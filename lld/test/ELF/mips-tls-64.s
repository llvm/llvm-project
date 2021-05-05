# REQUIRES: mips
# Check MIPS TLS 64-bit relocations handling.

# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux \
# RUN:         %p/Inputs/mips-tls.s -o %t.so.o
# RUN: ld.lld -shared %t.so.o -soname=t.so -o %t.so
# RUN: llvm-mc -filetype=obj -triple=mips64-unknown-linux %s -o %t.o

# RUN: echo "SECTIONS { \
# RUN:         . = 0x10000; .text : { *(.text) } \
# RUN:         . = 0x30000; .got  : { *(.got)  } \
# RUN:       }" > %t.script

# RUN: ld.lld %t.o %t.so -script %t.script -o %t.exe
# RUN: llvm-objdump -d -s -t --no-show-raw-insn %t.exe \
# RUN:   | FileCheck -check-prefix=DIS %s
# RUN: llvm-readobj -r -A %t.exe | FileCheck %s

# RUN: ld.lld -pie %t.o %t.so -script %t.script -o %t.pie
# RUN: llvm-objdump -d -s -t --no-show-raw-insn %t.pie \
# RUN:   | FileCheck -check-prefix=DIS %s
# RUN: llvm-readobj -r -A %t.pie | FileCheck %s

# RUN: ld.lld -shared %t.o %t.so -script %t.script -o %t-out.so
# RUN: llvm-objdump -d -s -t --no-show-raw-insn %t-out.so \
# RUN:   | FileCheck -check-prefix=DIS-SO %s
# RUN: llvm-readobj -r -A %t-out.so | FileCheck -check-prefix=SO %s

# DIS: 0000000000000000 l       .tdata  0000000000000000 loc
# DIS: 0000000000000000         *UND*   0000000000000000 foo
# DIS: 0000000000000004 g       .tdata  0000000000000000 bar

# DIS:      Contents of section .got:
# DIS-NEXT:  30000 00000000 00000000 80000000 00000000
# DIS-NEXT:  30010 00000000 00000000 ffffffff ffff9000
# DIS-NEXT:  30020 ffffffff ffff9004 00000000 00000000
# DIS-NEXT:  30030 00000000 00000000 00000000 00000001
# DIS-NEXT:  30040 00000000 00000000 00000000 00000001
# DIS-NEXT:  30050 ffffffff ffff8004

# DIS:      <__start>:
# DIS-NEXT:    addiu   $2, $3, -32712
# DIS-NEXT:    addiu   $2, $3, -32736
# DIS-NEXT:    addiu   $2, $3, -32696
# DIS-NEXT:    addiu   $2, $3, -32728
# DIS-NEXT:    addiu   $2, $3, -32680
# DIS-NEXT:    addiu   $2, $3, -32720

# CHECK:      Relocations [
# CHECK-NEXT:   Section (7) .rel.dyn {
# CHECK-NEXT:     0x30010 R_MIPS_TLS_TPREL64/R_MIPS_NONE/R_MIPS_NONE foo
# CHECK-NEXT:     0x30028 R_MIPS_TLS_DTPMOD64/R_MIPS_NONE/R_MIPS_NONE foo
# CHECK-NEXT:     0x30030 R_MIPS_TLS_DTPREL64/R_MIPS_NONE/R_MIPS_NONE foo
# CHECK-NEXT:   }
# CHECK-NEXT: ]
# CHECK:      Primary GOT {
# CHECK-NEXT:   Canonical gp value: 0x37FF0
# CHECK-NEXT:   Reserved entries [
# CHECK:        ]
# CHECK-NEXT:   Local entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Global entries [
# CHECK-NEXT:   ]
# CHECK-NEXT:   Number of TLS and multi-GOT entries: 9
#               ^-- -32736 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL64  foo
#               ^-- -32728 R_MIPS_TLS_GOTTPREL VA - 0x7000 loc
#               ^-- -32720 R_MIPS_TLS_GOTTPREL VA - 0x7000 bar
#               ^-- -32712 R_MIPS_TLS_GD       R_MIPS_TLS_DTPMOD64 foo
#               ^-- -32704                     R_MIPS_TLS_DTPREL64 foo
#               ^-- -32696 R_MIPS_TLS_LDM      1 loc
#               ^-- -32688                     0 loc
#               ^-- -32680 R_MIPS_TLS_GD       1 bar
#               ^-- -32672                     VA - 0x8000 bar

# DIS-SO:      Contents of section .got:
# DIS-SO-NEXT:  30000 00000000 00000000 80000000 00000000
# DIS-SO-NEXT:  30010 00000000 00000000 00000000 00000000
# DIS-SO-NEXT:  30020 00000000 00000004 00000000 00000000
# DIS-SO-NEXT:  30030 00000000 00000000 00000000 00000000
# DIS-SO-NEXT:  30040 00000000 00000000 00000000 00000000
# DIS-SO-NEXT:  30050 00000000 00000000

# SO:      Relocations [
# SO-NEXT:   Section (7) .rel.dyn {
# SO-NEXT:     0x30018 R_MIPS_TLS_TPREL64/R_MIPS_NONE/R_MIPS_NONE -
# SO-NEXT:     0x30038 R_MIPS_TLS_DTPMOD64/R_MIPS_NONE/R_MIPS_NONE -
# SO-NEXT:     0x30010 R_MIPS_TLS_TPREL64/R_MIPS_NONE/R_MIPS_NONE foo
# SO-NEXT:     0x30028 R_MIPS_TLS_DTPMOD64/R_MIPS_NONE/R_MIPS_NONE foo
# SO-NEXT:     0x30030 R_MIPS_TLS_DTPREL64/R_MIPS_NONE/R_MIPS_NONE foo
# SO-NEXT:     0x30020 R_MIPS_TLS_TPREL64/R_MIPS_NONE/R_MIPS_NONE bar
# SO-NEXT:     0x30048 R_MIPS_TLS_DTPMOD64/R_MIPS_NONE/R_MIPS_NONE bar
# SO-NEXT:     0x30050 R_MIPS_TLS_DTPREL64/R_MIPS_NONE/R_MIPS_NONE bar
# SO-NEXT:   }
# SO-NEXT: ]
# SO:      Primary GOT {
# SO-NEXT:   Canonical gp value: 0x37FF0
# SO-NEXT:   Reserved entries [
# SO:        ]
# SO-NEXT:   Local entries [
# SO-NEXT:   ]
# SO-NEXT:   Global entries [
# SO-NEXT:   ]
# SO-NEXT:   Number of TLS and multi-GOT entries: 9
#            ^-- -32736 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL64  foo
#            ^-- -32728 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL64  loc
#            ^-- -32720 R_MIPS_TLS_GOTTPREL R_MIPS_TLS_TPREL64  bar
#            ^-- -32712 R_MIPS_TLS_GD       R_MIPS_TLS_DTPMOD64 foo
#            ^-- -32704                     R_MIPS_TLS_DTPREL64 foo
#            ^-- -32696 R_MIPS_TLS_LDM      R_MIPS_TLS_DTPMOD64 loc
#            ^-- -32688                     0 loc
#            ^-- -32680 R_MIPS_TLS_GD       R_MIPS_TLS_DTPMOD64 bar
#            ^-- -32672                     R_MIPS_TLS_DTPREL64 bar

  .text
  .global  __start
__start:
  addiu $2, $3, %tlsgd(foo)     # R_MIPS_TLS_GD
  addiu $2, $3, %gottprel(foo)  # R_MIPS_TLS_GOTTPREL
  addiu $2, $3, %tlsldm(loc)    # R_MIPS_TLS_LDM
  addiu $2, $3, %gottprel(loc)  # R_MIPS_TLS_GOTTPREL
  addiu $2, $3, %tlsgd(bar)     # R_MIPS_TLS_GD
  addiu $2, $3, %gottprel(bar)  # R_MIPS_TLS_GOTTPREL

 .section .tdata,"awT",%progbits
 .global bar
loc:
 .word 0
bar:
 .word 0
