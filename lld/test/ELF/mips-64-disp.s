# REQUIRES: mips
# Check R_MIPS_GOT_DISP relocations against various kind of symbols.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         %p/Inputs/mips-pic.s -o %t.so.o
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.exe.o
# RUN: ld.lld %t.so.o -shared -soname=t.so -o %t.so
# RUN: ld.lld %t.exe.o %t.so -o %t.exe
# RUN: llvm-objdump --no-print-imm-hex -d -t --no-show-raw-insn %t.exe | FileCheck %s
# RUN: llvm-readelf -r -s -A %t.exe | FileCheck -check-prefix=GOT %s

# CHECK:      <__start>:
# CHECK-NEXT:    {{.*}}:  addiu   $2, $2, -32704
# CHECK-EMPTY:
# CHECK-NEXT: <b4>:
# CHECK-NEXT:    {{.*}}:  addiu   $2, $2, -32736
# CHECK-EMPTY:
# CHECK-NEXT: <b8>:
# CHECK-NEXT:    {{.*}}:  addiu   $2, $2, -32728
# CHECK-EMPTY:
# CHECK-NEXT: <b12>:
# CHECK-NEXT:    {{.*}}:  addiu   $2, $2, -32720
# CHECK-NEXT:    {{.*}}:  addiu   $2, $2, -32712

# GOT: Symbol table '.symtab'
# GOT: {{.*}} [[B04:[0-9a-f]+]] {{.*}} b4
# GOT: {{.*}} [[FOO:[0-9a-f]+]] {{.*}} foo
# GOT: {{.*}} [[B08:[0-9a-f]+]] {{.*}} b8
# GOT: {{.*}} [[B12:[0-9a-f]+]] {{.*}} b12

# GOT:      Primary GOT:
# GOT-NEXT:  Canonical gp value:
# GOT-EMPTY:
# GOT-NEXT:  Reserved entries:
# GOT-NEXT:  Address     Access          Initial Purpose
# GOT-NEXT:   {{.*}} -32752(gp) 0000000000000000 Lazy resolver
# GOT-NEXT:   {{.*}} -32744(gp) 8000000000000000 Module pointer (GNU extension)
# GOT-EMPTY:
# GOT-NEXT:  Local entries:
# GOT-NEXT:  Address     Access          Initial
# GOT-NEXT:   {{.*}} -32736(gp) [[FOO]]
# GOT-NEXT:   {{.*}} -32728(gp) [[B04]]
# GOT-NEXT:   {{.*}} -32720(gp) [[B08]]
# GOT-NEXT:   {{.*}} -32712(gp) [[B12]]
# GOT-EMPTY:
# GOT-NEXT:  Global entries:
# GOT-NEXT:  Address     Access          Initial         Sym.Val. Type Ndx Name
# GOT-NEXT:   {{.*}} -32704(gp) 0000000000000000 0000000000000000 FUNC UND foo1a

  .text
  .global  __start
__start:
  addiu   $v0,$v0,%got_disp(foo1a)            # R_MIPS_GOT_DISP
b4:
  addiu   $v0,$v0,%got_disp(foo)              # R_MIPS_GOT_DISP
b8:
  addiu   $v0,$v0,%got_disp(.text+4)          # R_MIPS_GOT_DISP
b12:
  addiu   $v0,$v0,%got_disp(.text+8)          # R_MIPS_GOT_DISP
  addiu   $v0,$v0,%got_disp(.text+12)         # R_MIPS_GOT_DISP

foo:
  nop
