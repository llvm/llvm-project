# REQUIRES: mips
# Check creation of GOT entries for global symbols in case of executable
# file linking. Symbols defined in DSO should get entries in the global part
# of the GOT. Symbols defined in the executable itself should get local GOT
# entries and does not need a row in .dynsym table.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         %S/Inputs/mips-dynamic.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux %s -o %t.o
# RUN: ld.lld %t.o %t.so -o %t.exe
# RUN: llvm-readelf --dyn-syms --symbols -A %t.exe | FileCheck %s

# CHECK: Symbol table '.dynsym'
# CHECK-NOT: bar

# CHECK: Symbol table '.symtab'
# CHECK: {{.*}}: 00000000            {{.*}} _foo
# CHECK: {{.*}}: [[BAR:[0-9a-f]+]] {{.*}} bar

# CHECK: Primary GOT:
# CHECK:  Local entries:
# CHECK-NEXT:    Address     Access  Initial
# CHECK-NEXT:     {{.*}} -32744(gp)  [[BAR]]
# CHECK-EMPTY:
# CHECK-NEXT:  Global entries:
# CHECK-NEXT:    Address     Access  Initial Sym.Val. Type    Ndx Name
# CHECK-NEXT:     {{.*}} -32740(gp) 00000000 00000000 NOTYPE  UND _foo

  .text
  .globl  __start
__start:
  lw      $t0,%got(bar)($gp)
  lw      $t0,%got(_foo)($gp)

.global bar
bar:
  .word 0
