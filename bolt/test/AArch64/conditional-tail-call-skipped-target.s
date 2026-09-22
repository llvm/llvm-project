## Preserve tail-call annotations on local trampolines, including reused ones.
## The indirect branch makes _start non-simple. Its conditional tail calls
## are redirected through a shared trampoline whose outgoing B must also be
## annotated as a tail call, even when the destination is excluded from layout.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,_start %s -o %t -nostdlib
# RUN: llvm-bolt %t -o %t.bolt --relax-exp --skip-funcs='^skipped$' \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-BOLT: BOLT-INFO: relaxed 1 calls with long thunks
# CHECK-BOLT: BOLT-INFO: 1 long thunks created

  .text
  .globl _start
  .type _start, %function
_start:
  cmp x0, #0
  b.eq skipped
  cmp x1, #0
  b.eq skipped
  ldr x16, [sp]
  br x16
  .size _start, .-_start

  .globl skipped
  .type skipped, %function
skipped:
  ret
  .size skipped, .-skipped

  .reloc 0, R_AARCH64_NONE

# CHECK-OUTPUT: Disassembly of section .text:
# CHECK-OUTPUT:      <_start>:
# CHECK-OUTPUT-NEXT:                  {{.*}} cmp x0, #0x0
# CHECK-OUTPUT-NEXT:                  {{.*}} b.eq 0x[[STUB:[0-9a-f]+]] <{{.*}}>
# CHECK-OUTPUT-NEXT:                  {{.*}} cmp x1, #0x0
# CHECK-OUTPUT-NEXT:                  {{.*}} b.eq 0x[[STUB]] <{{.*}}>
# CHECK-OUTPUT-NEXT:                  {{.*}} ldr x16, [sp]
# CHECK-OUTPUT-NEXT:                  {{.*}} br x16
# CHECK-OUTPUT-NEXT: [[STUB]]:        {{.*}} b 0x[[THUNK:[0-9a-f]+]] <__AArch64_forward_ADRPThunk_skipped_0>
# CHECK-OUTPUT:      <__AArch64_forward_ADRPThunk_skipped_0>:
# CHECK-OUTPUT-NEXT: [[THUNK]]:       {{.*}} adrp x16,
# CHECK-OUTPUT-NEXT:                  {{.*}} add x16, x16,
# CHECK-OUTPUT-NEXT:                  {{.*}} br x16
