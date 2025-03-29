! Test https://github.com/llvm/llvm-project/issues/75728 fix.
! RUN: bbc -emit-hlfir -o - -I nw %s | FileCheck %s

subroutine test()
  type t
    complex :: z
  end type
  type(t), target, save :: obj
  real, pointer :: p => obj%z%re
end subroutine
! CHECK-LABEL:   fir.global internal @_QFtestEp : !fir.box<!fir.ptr<f32>> {
! CHECK-NEXT:      %[[VAL_0:.*]] = fir.address_of(@_QFtestEobj) : !fir.ref<!fir.type<_QFtestTt{z:complex<f32>}>>
! CHECK-NEXT:      %[[VAL_1:.*]]:2 = hlfir.declare %[[VAL_0]] {fortran_attrs = #fir.var_attrs<target>, uniq_name = "_QFtestEobj"} : (!fir.ref<!fir.type<_QFtestTt{z:complex<f32>}>>) -> (!fir.ref<!fir.type<_QFtestTt{z:complex<f32>}>>, !fir.ref<!fir.type<_QFtestTt{z:complex<f32>}>>)
! CHECK-NEXT:      %[[VAL_2:.*]] = hlfir.designate %[[VAL_1]]#0{"z"}  real : (!fir.ref<!fir.type<_QFtestTt{z:complex<f32>}>>) -> !fir.ref<f32>
! CHECK-NEXT:      %[[VAL_3:.*]] = fir.embox %[[VAL_2]] : (!fir.ref<f32>) -> !fir.box<f32>
! CHECK-NEXT:      %[[VAL_4:.*]] = fir.rebox %[[VAL_3]] : (!fir.box<f32>) -> !fir.box<!fir.ptr<f32>>
! CHECK-NEXT:      fir.has_value %[[VAL_4]] : !fir.box<!fir.ptr<f32>>
