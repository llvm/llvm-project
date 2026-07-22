! RUN: %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: bbc -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

subroutine target_firstprivate_scalar
  implicit none
  integer :: x

  x = 42
  !$omp target firstprivate(x)
    x = x + 1
  !$omp end target
end subroutine target_firstprivate_scalar

! CHECK-LABEL: omp.private {type = firstprivate}
! CHECK-SAME:              @[[VAR_PRIVATIZER_SYM:.*]] : i32 copy {
! CHECK:         ^bb0(%[[ORIG:.*]]: !fir.ref<i32>, %[[PRIV:.*]]: !fir.ref<i32>):
! CHECK:           %[[VAL:.*]] = fir.load %[[ORIG]]
! CHECK:           hlfir.assign %[[VAL]] to %[[PRIV]]
! CHECK:           omp.yield(%[[PRIV]] : !fir.ref<i32>)
! CHECK:         }

! CHECK-LABEL: func.func @_QPtarget_firstprivate_scalar()
! CHECK:         %[[X_ALLOC:.*]] = fir.alloca i32 {bindc_name = "x", {{.*}}}
! CHECK:         %[[X_DECL:.*]]:2 = hlfir.declare %[[X_ALLOC]]

! CHECK:         omp.target {{.*}} private(
! CHECK-SAME:      @[[VAR_PRIVATIZER_SYM]] %[[X_DECL]]#0 -> %[[PRIV_ARG:.*]] [map_idx=0] : !fir.ref<i32>) {
! CHECK:           %[[PRIV_DECL:.*]]:2 = hlfir.declare %[[PRIV_ARG]]
! CHECK:           omp.terminator
! CHECK:         }

! CHECK:         return
! CHECK:       }
