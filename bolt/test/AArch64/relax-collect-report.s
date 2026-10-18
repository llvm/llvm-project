## Check the debug report that the relaxation survey prints. A call through a
## register has no target symbol, so the survey skips it and counts it. The
## direct call to B stays in the same cluster, so it is counted as intra-cluster.

# REQUIRES: system-linux
# REQUIRES: asserts

# RUN: %clang %cflags -Wl,-q -Wl,-e,A %s -o %t -nostdlib
# RUN: llvm-bolt %t -o %t.bolt --relax-exp -debug-only=longjmp 2>&1 \
# RUN:   | FileCheck %s

# CHECK: LongJmp: collect: no target symbol for call in A at layout offset 0x{{[0-9a-f]+}}
# CHECK: LongJmp: collect: collected 0 call(s) and 0 branch(es)
# CHECK-NEXT: LongJmp: collect: not considered for relaxation:
# CHECK-NEXT: LongJmp: collect:   0 function(s) not emitted or patched
# CHECK-NEXT: LongJmp: collect:   0 block(s) with no layout position
# CHECK-NEXT: LongJmp: collect:   1 call(s) with no target symbol
# CHECK-NEXT: LongJmp: collect:   0 branch(es) with no target symbol
# CHECK-NEXT: LongJmp: collect:   1 call(s) and 0 branch(es) inside one cluster, assumed in range
# CHECK-NEXT: LongJmp: collect:   widest intra-cluster reference: {{[0-9]+}} bytes; compare against the target's branch reach

  .text
  .globl A
  .type A, %function
A:
  bl B
  blr x1
  ret
  .size A, .-A

  .globl B
  .type B, %function
B:
  ret
  .size B, .-B

## Force relocation mode.
  .reloc 0, R_AARCH64_NONE
