! Verify that a large COMMON block array initialized on only a few elements by a
! DATA statement lowers to a compact constant global instead of an enormous
! `insertvalue` chain (which is pathologically slow to generate, see
! https://github.com/llvm/llvm-project/issues/209393).

! RUN: %flang_fc1 -emit-llvm %s -o - | FileCheck %s

block data
  integer i(55000)
  common /b/ i
  data (i(j), j = 1, 2) / 7, 77 /
end block data

! CHECK: @b_ = {{.*}}global { [55000 x i32] } { [55000 x i32] [i32 7, i32 77, i32 0
! CHECK-NOT: insertvalue
