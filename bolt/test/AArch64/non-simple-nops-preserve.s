# This test checks that the nop instruction is preserved before the
# ADRRelaxation pass. Otherwise the ADRRelaxation pass would
# fail to replace nop+adr to adrp+add for non-simple function.

# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown \
# RUN:   %s -o %t.o
# RUN: %clang %cflags %t.o -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --adr-relaxation=true \
# RUN:   --print-cfg --print-only=_start | FileCheck %s
# RUN: llvm-objdump -d -j .text %t.bolt | \
# RUN:   FileCheck --check-prefix=DISASMCHECK %s

# CHECK: Binary Function "_start" after building cfg
# CHECK: IsSimple : 0

# DISASMCHECK: {{.*}}<_start>:
# DISASMCHECK-NEXT: adrp
# DISASMCHECK-NEXT: add
# DISASMCHECK-NEXT: adr

  .text
  .align 4
  .global test
  .type test, %function
test:
  ret
  .size test, .-test

  .global _start
  .type _start, %function
_start:
  nop
  adr x0, test
  adr x1, 1f
1:
  mov x1, x0
  br x0
  .size _start, .-_start
