// This test checks that the constant island is aligned after BOLT tool.

# RUN: split-file %s %t

// For the first test case, in case the nop before .Lci will be removed
// the pointer to exit function won't be aligned and the test will fail.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %t/xword_align.s -o %t_xa.o
# RUN: %clang %cflags -fPIC -pie %t_xa.o -o %t_xa.exe -Wl,-q \
# RUN:    -nostartfiles -nodefaultlibs -Wl,-z,notext
# RUN: llvm-bolt %t_xa.exe -o %t_xa.bolt --use-old-text=0 --lite=0 \
# RUN:    --trap-old-code
# RUN: llvm-objdump -d --disassemble-symbols='$d' %t_xa.bolt | FileCheck %s

// For the second and third test cases, we want to set an alignment based
// on various heuristics.

# RUN: %clang %cflags -pie %t/page_align.s -o %t_pa.exe -Wl,-q \
# RUN:    -Wl,--init=_foo -Wl,--fini=_foo
# RUN: llvm-bolt %t_pa.exe -o %t_pa.bolt
# RUN: llvm-objdump -t %t_pa.exe | grep _const_island
# RUN: llvm-objdump -t %t_pa.bolt | grep _const_island | FileCheck %s \
# RUN:    --check-prefix=PAGE

# RUN: %clang %cflags -pie %t/64B_align.s -o %t_64B.exe -Wl,-q \
# RUN:    -Wl,--init=_foo -Wl,--fini=_foo
# RUN: llvm-bolt %t_64B.exe -o %t_64B.bolt
# RUN: llvm-objdump -t %t_64B.exe | grep _const_island
# RUN: llvm-objdump -t %t_64B.bolt | grep _const_island | FileCheck %s \
# RUN:    --check-prefix=64BYTE

;--- xword_align.s
.text
.align 4
.global
.type dummy, %function
dummy:
  add x0, x0, #1
  ret

.global
.type exitOk, %function
exitOk:
  mov x0, #0
  ret

.global _start
.type _start, %function
_start:
  adrp x0, .Lci
  ldr x0, [x0, #:lo12:.Lci]
  blr x0
  mov x0, #1
  ret
  nop
# CHECK: {{0|8}} <$d>:
.Lci:
  .xword exitOk
  .xword 0

;--- page_align.s
  .text
  .global _foo
  .type _foo, %function
_foo:
  ret

  .text
  .global _const_island
  .align 12
# PAGE: {{[0-9a-f]*}}000 g
_const_island:
  .rept 0x25100
    .byte 0xbb
  .endr

  .global _start
  .type _start, %function
_start:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE

;--- 64B_align.s
  .text
  .global _foo
  .type _foo, %function
_foo:
  ret

  .text
  .global _const_island
  .align 6
# 64BYTE: {{[0-9a-f]*}}{{0|4|8|c}}0 g
_const_island:
  .rept 0x2048
    .byte 0xbb
  .endr

  .global _start
  .type _start, %function
_start:
  ret

  # Dummy relocation to force relocation mode
  .reloc 0, R_AARCH64_NONE
