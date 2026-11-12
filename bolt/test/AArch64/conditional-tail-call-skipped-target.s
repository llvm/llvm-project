## Check that a conditional tail call's local trampoline has a tail-call
## annotation on its outgoing branch. The target is skipped and absent from the
## output layout, so cluster relaxation needs that annotation for a long thunk.
## The unresolved indirect branch makes the caller non-simple.

# REQUIRES: system-linux

# RUN: %clang %cflags -Wl,-q -Wl,-e,non_simple %s -o %t -nostdlib
# RUN: llvm-bolt %t -o %t.bolt --relax-exp --skip-funcs='^skipped$' \
# RUN:   --print-cfg --print-only=non_simple \
# RUN:   | FileCheck %s --check-prefix=CHECK-BOLT
# RUN: llvm-objdump -d %t.bolt | FileCheck %s --check-prefix=CHECK-OUTPUT

# CHECK-BOLT: Binary Function "non_simple" after building cfg
# CHECK-BOLT: IsSimple    : 0
# CHECK-BOLT: BOLT-INFO: relaxed 1 calls with long thunks
# CHECK-BOLT: BOLT-INFO: 1 long thunks created

  .text
  .globl non_simple
  .type non_simple, %function
non_simple:
  cmp x0, #0
  b.eq skipped
  br x1
  .size non_simple, .-non_simple

  .globl skipped
  .type skipped, %function
skipped:
  ret
  .size skipped, .-skipped

.reloc 0, R_AARCH64_NONE

# CHECK-OUTPUT: Disassembly of section .text:
#
# CHECK-OUTPUT:      <non_simple>:
# CHECK-OUTPUT-NEXT:           {{.*}} cmp x0, #0x0
# CHECK-OUTPUT-NEXT:           {{.*}} b.eq 0x[[STUB:[0-9a-f]+]] <{{.*}}>
# CHECK-OUTPUT-NEXT:           {{.*}} br x1
# CHECK-OUTPUT-NEXT: [[STUB]]: {{.*}} b 0x{{[0-9a-f]+}} <__AArch64_forward_ADRPThunk_skipped_0>
#
# CHECK-OUTPUT:      <__AArch64_forward_ADRPThunk_skipped_0>:
# CHECK-OUTPUT-NEXT:   {{.*}} adrp x16,
# CHECK-OUTPUT-NEXT:   {{.*}} add x16, x16,
# CHECK-OUTPUT-NEXT:   {{.*}} br x16
