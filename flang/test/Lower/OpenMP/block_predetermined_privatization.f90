! Fixes a bug when a block variable is marked as pre-determined private. In such
! case, we can simply ignore privatizing that symbol within the context of the
! currrent OpenMP construct since the "private" allocation for the symbol will
! be emitted within the nested block anyway.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine block_predetermined_privatization
  implicit none
  integer :: i

  !$omp parallel
  do i=1,10
    block
      integer :: j
      do j=1,10
      end do
    end block
  end do
  !$omp end parallel
end subroutine

! CHECK-LABEL: func.func @_QPblock_predetermined_privatization() {
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Ei"}
! CHECK:         omp.parallel private(@{{.*}}Ei_private_i32 %[[I_DECL]]#0 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:           fir.do_loop {{.*}} {
! Verify that `j` is allocated whithin the same scope of its block (i.e. inside
! the `parallel` loop).
! CHECK:             fir.alloca i32 {bindc_name = "j", {{.*}}}
! CHECK:           }
! CHECK:         }
! CHECK:       }
