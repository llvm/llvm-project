!RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

module foo
    implicit none
    contains
    subroutine test(I,A)
      implicit none
      real(4), optional, intent(inout) :: A(:)
      integer(kind=4), intent(in) :: I

     !$omp target data map(to: A) if (I>0)
     !$omp end target data

    end subroutine test
end module foo

! CHECK-LABEL:   func.func @_QMfooPtest(
! CHECK-SAME:                           %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "i"},
! CHECK-SAME:                           %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a", fir.optional}) {
! CHECK:           %[[VAL_2:.*]] = fir.alloca !fir.box<!fir.array<?xf32>>
! CHECK:           %[[VAL_3:.*]]:2 = hlfir.declare %[[VAL_1]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<intent_inout, optional>, uniq_name = "_QMfooFtestEa"} : (!fir.box<!fir.array<?xf32>>, !fir.dscope) -> (!fir.box<!fir.array<?xf32>>, !fir.box<!fir.array<?xf32>>)
! CHECK:           %{{.*}} = fir.is_present %{{.*}}#1 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:           %{{.*}}:5 = fir.if %{{.*}}
! CHECK:           %[[VAL_4:.*]] = fir.is_present %[[VAL_3]]#1 : (!fir.box<!fir.array<?xf32>>) -> i1
! CHECK:           fir.if %[[VAL_4]] {
! CHECK:             fir.store %[[VAL_3]]#1 to %[[VAL_2]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:           }
