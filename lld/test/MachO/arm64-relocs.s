# REQUIRES: aarch64
# RUN: llvm-mc -filetype=obj -triple=arm64-apple-darwin %s -o %t.o
# RUN: %lld -dylib -arch arm64 -lSystem -o %t %t.o
# RUN: llvm-objdump --syms %t > %t.objdump
# RUN: llvm-objdump --no-print-imm-hex --macho -d --section=__const %t >> %t.objdump
# RUN: FileCheck %s < %t.objdump

# CHECK-LABEL: SYMBOL TABLE:
# CHECK-DAG:   [[#%x,PTR_1:]] l     O __DATA_CONST,__const _ptr_1
# CHECK-DAG:   [[#%x,PTR_2:]] l     O __DATA_CONST,__const _ptr_2
# CHECK-DAG:   [[#%x,BAR:]]   g     F __TEXT,__text _bar
# CHECK-DAG:   [[#%x,BAZ:]]   g     O __DATA,__data _baz

# CHECK-LABEL: _foo:
## BRANCH26 relocations are 4-byte aligned, so 123 is truncated to 120
# CHECK-NEXT:  bl   0x[[#BAR+120]]
## PAGE21 relocations are aligned to 4096 bytes
# CHECK-NEXT:  adrp x2, [[#]] ; 0x[[#BAZ+4096-128]]
# CHECK-NEXT:  ldr  x2, [x2, #128]
# CHECK-NEXT:  adrp x3, 8 ; 0x8000
# CHECK-NEXT:  ldr  q0, [x3, #144]
# CHECK-NEXT:  ret

# CHECK-LABEL: Contents of (__DATA_CONST,__const) section
# CHECK:       [[#PTR_1]]  {{0*}}[[#BAZ]]     00000000 00000000 00000000
# CHECK:       [[#PTR_2]]  {{0*}}[[#BAZ+123]] 00000000 00000000 00000000

.text
.globl _foo, _bar, _baz, _quux
.p2align 2
_foo:
  ## Generates ARM64_RELOC_BRANCH26 and ARM64_RELOC_ADDEND
  bl _bar + 123
  ## Generates ARM64_RELOC_PAGE21 and ADDEND
  adrp x2, _baz@PAGE + 4097
  ## Generates ARM64_RELOC_PAGEOFF12
  ldr x2, [x2, _baz@PAGEOFF]

  ## Generates ARM64_RELOC_PAGE21
  adrp x3, _quux@PAGE
  ## Generates ARM64_RELOC_PAGEOFF12 with internal slide 4
  ldr q0, [x3, _quux@PAGEOFF]
  ret

.p2align 2
_bar:
  ret

.data
.space 128
_baz:
.space 1

.p2align 4
_quux:
.quad 0
.quad 80

.section __DATA_CONST,__const
## These generate ARM64_RELOC_UNSIGNED symbol relocations. llvm-mc seems to
## generate UNSIGNED section relocations only for compact unwind sections, so
## those relocations are being tested in compact-unwind.s.
_ptr_1:
  .quad _baz
  .space 8
_ptr_2:
  .quad _baz + 123
  .space 8

.subsections_via_symbols
