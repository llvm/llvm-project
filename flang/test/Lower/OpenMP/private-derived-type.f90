! RUN: %flang_fc1 -emit-hlfir -fopenmp -mmlir --enable-delayed-privatization-staging=true -o - %s | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp --enable-delayed-privatization-staging=true -o - %s | FileCheck %s

subroutine s4
  type y3
    integer,allocatable::x
  end type y3
  type(y3)::v
  !$omp parallel
  !$omp do private(v)
  do i=1,10
    v%x=1
  end do
  !$omp end do
  !$omp end parallel
end subroutine s4

! CHECK:  omp.private {type = private} @[[DERIVED_PRIV:.*]] : !fir.type<{{.*}}y3{x:!fir.box<!fir.heap<i32>>}> init {
! CHECK:             %[[VAL_25:.*]] = fir.embox %[[VAL_23:.*]] : (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:             %[[VAL_26:.*]] = fir.address_of
! CHECK:             %[[VAL_27:.*]] = arith.constant 8 : i32
! CHECK:             %[[VAL_28:.*]] = fir.convert %[[VAL_25]] : (!fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
! CHECK:             %[[VAL_29:.*]] = fir.convert %[[VAL_26]] : (!fir.ref<!fir.char<1,{{.*}}>>) -> !fir.ref<i8>
!                    Check we do call FortranAInitialize on the derived type
! CHECK:             fir.call @_FortranAInitialize(%[[VAL_28]], %[[VAL_29]], %[[VAL_27]]) fastmath<contract> : (!fir.box<none>, !fir.ref<i8>, i32) -> ()
! CHECK:  }

! CHECK-LABEL:   func.func @_QPs4() {
!                  Example of how the lowering for regular derived type variables:
! CHECK:           %[[VAL_DERIVED_ALLOCA:.*]] = fir.alloca !fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}> {bindc_name = "v", uniq_name = "_QFs4Ev"}
! CHECK:           %[[VAL_DERIVED_DECLARE:.*]]:2 = hlfir.declare %[[VAL_DERIVED_ALLOCA]] {{.*}}
! CHECK:           %[[ADDR:.*]] = fir.address_of(@_QQ_QFs4Ty3.DerivedInit) : !fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:           fir.copy %[[ADDR]] to %[[VAL_DERIVED_DECLARE]]#0 no_overlap : {{.*}}
! CHECK:           omp.parallel {
! CHECK:             omp.wsloop private(@[[DERIVED_PRIV]] %{{.*}}#0 -> %{{.*}}, @{{.*}} %{{.*}}#0 -> %{{.*}} : !fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>, !fir.ref<i32>) {
! CHECK:           }
! CHECK:           %[[VAL_39:.*]] = fir.embox %[[VAL_DERIVED_DECLARE]]#0 : (!fir.ref<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>
! CHECK:           %[[VAL_40:.*]] = fir.convert %[[VAL_39]] : (!fir.box<!fir.type<_QFs4Ty3{x:!fir.box<!fir.heap<i32>>}>>) -> !fir.box<none>
!                  Check the derived type is destroyed
! CHECK:           fir.call @_FortranADestroy(%[[VAL_40]]) fastmath<contract> : (!fir.box<none>) -> ()
! CHECK:           return
! CHECK:         }
