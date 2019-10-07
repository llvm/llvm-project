# REQUIRES: mips
# Check less-significant bit setup for microMIPS PLT.

# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %S/Inputs/mips-dynamic.s -o %t-dso.o
# RUN: ld.lld %t-dso.o -shared -soname=t.so -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips-unknown-linux \
# RUN:         -mattr=micromips %s -o %t-exe.o
# RUN: echo "SECTIONS { \
# RUN:         . = 0x20000;  .text ALIGN(0x100) : { *(.text) } \
# RUN:         . = 0x20300;  .plt : { *(.plt) } \
# RUN:       }" > %t.script
# RUN: ld.lld %t-exe.o %t.so --script %t.script -o %t.exe
# RUN: llvm-readelf --symbols --dyn-syms -A %t.exe | FileCheck %s
# RUN: llvm-objdump -d -mattr=micromips --no-show-raw-insn %t.exe \
# RUN:   | FileCheck --check-prefix=ASM %s

# CHECK: Symbol table '.dynsym'
# CHECK:    Num:    Value  Size Type    Bind   Vis                    Ndx Name
# CHECK:      1: 00020321     0 FUNC    GLOBAL DEFAULT [<other: 0x88>] UND foo0

# CHECK: Symbol table '.symtab'
# CHECK:    Num:    Value  Size Type    Bind   Vis                    Ndx Name
# CHECK:      1: 00020210     0 NOTYPE  LOCAL  HIDDEN [<other: 0x82>]   8 foo
# CHECK:      4: 00020200     0 NOTYPE  GLOBAL DEFAULT [<other: 0x80>]   8 __start
# CHECK:      5: 00020320     0 FUNC    GLOBAL DEFAULT [<other: 0x88>] UND foo0

# CHECK: Primary GOT:
# CHECK:  Local entries:
# CHECK:    Address     Access  Initial
# CHECK:            -32744(gp) 00020211

# CHECK: PLT GOT:
# CHECK:  Entries:
# CHECK:    Address  Initial Sym.Val. Type    Ndx Name
# CHECK:            00020301 00020321 FUNC    UND foo0

# ASM:      __start:
# ASM-NEXT:    20200:  lw      $8, -32744($gp)
# ASM-NEXT:            addi    $8, $8, 529
# ASM-NEXT:            lui     $8, 2
# ASM-NEXT:            addi    $8, $8, 801
#
# ASM:      foo:
# ASM-NEXT:    20210:  jal     131872

  .text
  .set micromips
  .global foo
  .hidden foo
  .global __start
__start:
  lw    $t0,%got(foo)($gp)
  addi  $t0,$t0,%lo(foo)
  lui   $t0,%hi(foo0)
  addi  $t0,$t0,%lo(foo0)
foo:
  jal   foo0
