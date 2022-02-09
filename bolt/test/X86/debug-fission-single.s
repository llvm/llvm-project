# Checks debug fission support in BOLT

# REQUIRES: system-linux

# RUN: llvm-mc -g \
# RUN:   --filetype=obj \
# RUN:   --triple x86_64-unknown-unknown \
# RUN:   --split-dwarf-file=debug-fission-simple.dwo \
# RUN:   %p/Inputs/debug-fission-simple.s \
# RUN:   -o %t.o
# RUN: %clangxx %cxxflags -g \
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
# RUN: not llvm-dwarfdump --show-form --verbose --debug-info %T/debug-fission-simple.dwo0.dwo -o %tAddrIndexTest 2> /dev/null
# RUN: cat %tAddrIndexTest | grep DW_FORM_GNU_addr_index | FileCheck %s --check-prefix=CHECK-ADDR-INDEX
# RUN: llvm-dwarfdump --show-form --verbose   --debug-addr  %t.bolt.1.exe | FileCheck %s --check-prefix=CHECK-ADDR-SEC

# CHECK-NOT: warning: DWARF unit from offset {{.*}} incl. to offset {{.*}} excl. tries to read DIEs at offset {{.*}}

# CHECK-ADDR-INDEX: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000000)
# CHECK-ADDR-INDEX: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000000)
# CHECK-ADDR-INDEX: DW_AT_low_pc [DW_FORM_GNU_addr_index]	(indexed (00000000)

# CHECK-ADDR-SEC: .debug_addr contents:
# CHECK-ADDR-SEC: 0x00000000: Addrs: [
# CHECK-ADDR-SEC: 0x0000000000601000

# RUN: llvm-bolt %t.exe --reorder-blocks=reverse -update-debug-sections -dwarf-output-path=%T -o %t.bolt.2.exe --write-dwp=true
# RUN: not llvm-dwarfdump --show-form --verbose --debug-info %t.bolt.2.exe.dwp -o %tAddrIndexTestDwp 2> /dev/null
# RUN: cat %tAddrIndexTestDwp | FileCheck %s --check-prefix=CHECK-DWP-DEBUG

# CHECK-DWP-DEBUG: DW_TAG_compile_unit [1] *
# CHECK-DWP-DEBUG:  DW_AT_producer [DW_FORM_GNU_str_index]  (indexed (0000000a) string = "clang version 13.0.0")
# CHECK-DWP-DEBUG:  DW_AT_language [DW_FORM_data2]  (DW_LANG_C_plus_plus)
# CHECK-DWP-DEBUG:  DW_AT_name [DW_FORM_GNU_str_index]  (indexed (0000000b) string = "foo")
# CHECK-DWP-DEBUG:  DW_AT_GNU_dwo_name [DW_FORM_GNU_str_index]  (indexed (0000000c) string = "foo")
# CHECK-DWP-DEBUG:  DW_AT_GNU_dwo_id [DW_FORM_data8]  (0x06105e732fad3796)


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
