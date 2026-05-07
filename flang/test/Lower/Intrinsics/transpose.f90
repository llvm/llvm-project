! RUN: %flang_fc1 -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL: func @_QPtranspose_test(
! CHECK-SAME: %[[mat:.*]]: !fir.ref<!fir.array<2x3xf32>>{{.*}})
subroutine transpose_test(mat)
   real :: mat(2,3)
   call bar_transpose_test(transpose(mat))
! CHECK: %[[matDecl:.*]]:2 = hlfir.declare %[[mat]](%{{.*}}) {{.*}}{uniq_name = "_QFtranspose_testEmat"}
! CHECK: %[[result:.*]] = hlfir.transpose %[[matDecl]]#0 : (!fir.ref<!fir.array<2x3xf32>>) -> !hlfir.expr<3x2xf32>
! CHECK: %[[shape:.*]] = hlfir.shape_of %[[result]] : (!hlfir.expr<3x2xf32>) -> !fir.shape<2>
! CHECK: %[[assoc:.*]]:3 = hlfir.associate %[[result]](%[[shape]]) {adapt.valuebyref}
! CHECK: fir.call @_QPbar_transpose_test(%[[assoc]]#0)
! CHECK: hlfir.end_associate %[[assoc]]#1, %[[assoc]]#2
! CHECK: hlfir.destroy %[[result]] : !hlfir.expr<3x2xf32>
end subroutine
