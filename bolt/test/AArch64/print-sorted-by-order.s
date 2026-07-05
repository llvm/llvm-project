# Check that --print-sorted-by-order=<ascending/descending> option works properly in llvm-bolt
#
# RUN: llvm-mc -filetype=obj -triple aarch64-unknown-unknown %s -o %t.o
# RUN: %clang %cflags -fPIC -pie %t.o -o %t.exe -nostdlib -Wl,-q
# RUN: link_fdata %s %t.o %t.fdata
# RUN: llvm-bolt %t.exe -o %t.bolt --print-sorted-by=all --print-sorted-by-order=ascending \
# RUN:   --data %t.fdata | FileCheck %s -check-prefix=CHECK-ASCEND
# RUN: llvm-bolt %t.exe -o %t.bolt --print-sorted-by=all --print-sorted-by-order=descending \
# RUN:   --data %t.fdata | FileCheck %s -check-prefix=CHECK-DESCEND

# CHECK-ASCEND: BOLT-INFO: top functions sorted by dyno stats are:
# CHECK-ASCEND-NEXT: bar
# CHECK-ASCEND-NEXT: foo
# CHECK-DESCEND: BOLT-INFO: top functions sorted by dyno stats are:
# CHECK-DESCEND-NEXT: foo
# CHECK-DESCEND-NEXT: bar

  .text
  .align 4
  .global bar
  .type bar, %function
bar:
  mov w0, wzr
  ret

  .global foo
  .type foo, %function
foo:
# FDATA: 1 foo 0 1 bar 0 0 1
  bl bar
  ret
