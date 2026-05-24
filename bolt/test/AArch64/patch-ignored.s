## Check that llvm-bolt patches functions that are getting ignored after their
## CFG was constructed.

# RUN: %clang %cflags %s -o %t.exe -Wl,-q
# RUN: llvm-bolt %t.exe -o %t.bolt --force-patch 2>&1 | FileCheck %s
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-OBJDUMP

  .text

## The function is too small to be patched and BOLT is forced to ignore it under
## --force-patch. Check that the reference to _start is updated.
# CHECK: BOLT-WARNING: failed to patch entries in unpatchable
	.globl unpatchable
  .type unpatchable, %function
unpatchable:
  .cfi_startproc
# CHECK-OBJDUMP:      <unpatchable>:
# CHECK-OBJDUMP-NEXT: bl {{.*}} <_start>
  bl _start
  ret
  .cfi_endproc
  .size unpatchable, .-unpatchable

  .globl _start
  .type _start, %function
_start:
  .cfi_startproc
  cmp  x0, 1
  b.eq  .L0
.L0:
  ret  x30
  .cfi_endproc
  .size _start, .-_start
