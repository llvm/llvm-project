## Checks debug fission support in BOLT

# REQUIRES: system-linux

# RUN: llvm-mc -g \
# RUN:   --filetype=obj \
# RUN:   --triple x86_64-unknown-unknown \
# RUN:   --split-dwarf-file=debug-fission-simple.dwo \
# RUN:   %p/Inputs/debug-fission-simple.s \
# RUN:   -o %t.o
# RUN: %clangxx %cxxflags -no-pie -g \
# RUN:   -Wl,--gc-sections,-q,-nostdlib \
# RUN:   -Wl,--undefined=_Z6_startv \
# RUN:   -nostartfiles \
# RUN:   -Wl,--script=%p/Inputs/debug-fission-script.txt \
# RUN:   %t.o -o %t.exe
# RUN: llvm-bolt %t.exe \
# RUN:   --reorder-blocks=reverse \
# RUN:   --update-debug-sections \
# RUN:   --dwarf-output-path=%T \
# RUN:   -o %t.bolt.1.exe 2>&1 | FileCheck %s
# RUN: llvm-dwarfdump --show-form --verbose --debug-ranges %t.bolt.1.exe &> %tAddrIndexTest
# RUN: llvm-dwarfdump --show-form --verbose --debug-info %T/debug-fission-simple.dwo0.dwo >> %tAddrIndexTest
# RUN: cat %tAddrIndexTest | FileCheck %s --check-prefix=CHECK-DWO-DWO
# RUN: llvm-dwarfdump --show-form --verbose   --debug-addr  %t.bolt.1.exe | FileCheck %s --check-prefix=CHECK-ADDR-SEC

# CHECK-NOT: warning: DWARF unit from offset {{.*}} incl. to offset {{.*}} excl. tries to read DIEs at offset {{.*}}

# CHECK-DWO-DWO: 00000010
# CHECK-DWO-DWO: 00000010
# CHECK-DWO-DWO: DW_TAG_subprogram
# CHECK-DWO-DWO-NEXT: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000001)
# CHECK-DWO-DWO-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x00000031)
# CHECK-DWO-DWO: DW_TAG_subprogram
# CHECK-DWO-DWO-NEXT: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000002)
# CHECK-DWO-DWO-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x00000012)
# CHECK-DWO-DWO: DW_TAG_subprogram
# CHECK-DWO-DWO-NEXT: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000003)
# CHECK-DWO-DWO-NEXT: DW_AT_high_pc [DW_FORM_data4]	(0x0000001d)

# CHECK-ADDR-SEC: .debug_addr contents:
# CHECK-ADDR-SEC: 0x00000000: Addrs: [
# CHECK-ADDR-SEC: 0x0000000000601000

//clang++ -ffunction-sections -fno-exceptions -g -gsplit-dwarf=split -S debug-fission-simple.cpp -o debug-fission-simple.s
static int foo = 2;
int doStuff(int val) {
  if (val == 5)
    val += 1 + foo;
  else
    val -= 1;
  return val;
}

int doStuff2(int val) {
  return val += 3;
}

int main(int argc, const char** argv) {
  return doStuff(argc);
}
