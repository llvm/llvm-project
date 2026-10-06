## The text alignment allowance consumes the branch safety margin at the
## 126 MiB cluster-size ceiling. A call outside the emitted layout adds a
## long thunk, so the pass warns and still completes.

# REQUIRES: system-linux

# RUN: %clang %cflags -shared -Wl,-q %s -o %t
# RUN: llvm-bolt %t -o %t.bolt --relax-exp \
# RUN:   --max-cluster-size=132120576 --align-text=2097152 > %t.log 2>&1
# RUN: FileCheck %s < %t.log

# CHECK: BOLT-INFO: built 1 function fragment cluster(s)
# CHECK: BOLT-INFO:   12 thunk bytes
# CHECK: BOLT-WARNING: cluster 0: 12 thunk bytes plus 2097152 bytes
# CHECK-SAME: for text alignment exceed the 2097152-byte branch safety margin

  .text
  .weak external_func
  .globl A
  .type A, %function
A:
  bl external_func
  ret
  .size A, .-A
