// This test checks that the constant island is aligned after BOLT tool.
// In case the nop before .Lci will be removed the pointer to exit function
// won't be alinged and the test will fail.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -Wl,-q \
# RUN:    -nostartfiles -nodefaultlibs -lc
# RUN: llvm-bolt %t.exe -o %t.bolt -use-old-text=0 -lite=0 -trap-old-code
# RUN: llvm-objdump -d --disassemble-symbols='$d' %t.bolt | FileCheck %s

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
  bl exit

.global _start
.type _start, %function
_start:
  adrp x0, .Lci
  ldr x0, [x0, #:lo12:.Lci]
  blr x0
  mov x1, #1
  bl exit
  nop
# CHECK: {{0|8}} <$d>:
.Lci:
  .xword exitOk
  .xword 0
