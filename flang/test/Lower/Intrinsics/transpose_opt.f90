! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func.func @_QPtranspose_test(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.array<2x3xf32>>{{.*}})
subroutine transpose_test(mat)
   real :: mat(2,3)
   call bar_transpose_test(transpose(mat))
! CHECK: hlfir.declare %[[VAL_0]](%{{.*}}) {{.*}}{uniq_name = "_QFtranspose_testEmat"}
! CHECK: hlfir.transpose {{.*}} : (!fir.ref<!fir.array<2x3xf32>>) -> !hlfir.expr<3x2xf32>
! CHECK: fir.call @_QPbar_transpose_test
! CHECK-NOT: @_FortranATranspose
end subroutine

! CHECK-LABEL: func.func @_QPtranspose_allocatable_test(
! CHECK-SAME: %[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?x?xf32>>>>{{.*}})
subroutine transpose_allocatable_test(mat)
  real, allocatable :: mat(:,:)
  mat = transpose(mat)
! Verify that hlfir.transpose is used (not explicit loops or runtime call)
! CHECK: hlfir.declare %[[VAL_0]] {{.*}}{fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFtranspose_allocatable_testEmat"}
! CHECK: hlfir.transpose {{.*}} : (!fir.box<!fir.heap<!fir.array<?x?xf32>>>) -> !hlfir.expr<?x?xf32>
! CHECK: hlfir.assign {{.*}} realloc
! CHECK: hlfir.destroy {{.*}} : !hlfir.expr<?x?xf32>
! CHECK-NOT: @_FortranATranspose
end subroutine
