! RUN: bbc -emit-hlfir -fopenmp -o - %s | FileCheck %s

subroutine sub(imax, jmax, x, y)
  integer, intent(in) :: imax, jmax
  real, intent(in), dimension(1:imax, 1:jmax) :: x, y

  integer :: i, j, ii

  ! collapse(2) is needed to reproduce the issue
  !$omp parallel do collapse(2)
  do j = 1, jmax
    do i = 1, imax
      do  ii = 1, imax ! note that this loop is not collapsed
        if (x(i,j) < y(ii,j)) then
          ! exit needed to force unstructured control flow
          exit
        endif
      enddo
    enddo
  enddo
end subroutine sub

! this is testing that we don't crash generating code for this: in particular
! that all blocks are terminated

! CHECK-LABEL:   func.func @_QPsub(
! CHECK-SAME:                      %[[VAL_0:.*]]: !fir.ref<i32> {fir.bindc_name = "imax"},
! CHECK-SAME:                      %[[VAL_1:.*]]: !fir.ref<i32> {fir.bindc_name = "jmax"},
! CHECK-SAME:                      %[[VAL_2:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "x"},
! CHECK-SAME:                      %[[VAL_3:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "y"}) {
! [...]
! CHECK:             omp.wsloop {
! CHECK-NEXT:          omp.loop_nest (%[[VAL_53:.*]], %[[VAL_54:.*]]) : i32 = ({{.*}}) to ({{.*}}) inclusive step ({{.*}}) {
! [...]
! CHECK:                 cf.br ^bb1
! CHECK:               ^bb1:
! CHECK:                 cf.br ^bb2
! CHECK:               ^bb2:
! [...]
! CHECK:                 cf.br ^bb3
! CHECK:               ^bb3:
! [...]
! CHECK:                 %[[VAL_63:.*]] = arith.cmpi sgt, %{{.*}}, %{{.*}} : i32
! CHECK:                 cf.cond_br %[[VAL_63]], ^bb4, ^bb7
! CHECK:               ^bb4:
! [...]
! CHECK:                 %[[VAL_76:.*]] = arith.cmpf olt, %{{.*}}, %{{.*}} fastmath<contract> : f32
! CHECK:                 cf.cond_br %[[VAL_76]], ^bb5, ^bb6
! CHECK:               ^bb5:
! CHECK:                 cf.br ^bb7
! CHECK:               ^bb6:
! [...]
! CHECK:                 cf.br ^bb3
! CHECK:               ^bb7:
! CHECK:                 omp.yield
! CHECK:               }
! CHECK:             }
! CHECK:             omp.terminator
! CHECK:           }
! CHECK:           cf.br ^bb1
! CHECK:         ^bb1:
! CHECK:           return
! CHECK:         }
