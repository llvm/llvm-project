! When a block variable is marked as implicit private, we can simply ignore
! privatizing that symbol within the context of the currrent OpenMP construct
! since the "private" allocation for the symbol will be emitted within the nested
! block anyway.

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s

subroutine block_implicit_privatization
  implicit none
  integer :: i

  !$omp task
  do i=1,10
    block
      integer :: j
      j = 0
    end block
  end do
  !$omp end task
end subroutine

! CHECK-LABEL: func.func @_QPblock_implicit_privatization() {
! CHECK:         %[[I_DECL:.*]]:2 = hlfir.declare %{{.*}} {uniq_name = "{{.*}}Ei"}
! CHECK:         omp.task private(@{{.*}}Ei_private_i32 %[[I_DECL]]#0 -> %{{.*}} : !fir.ref<i32>) {
! CHECK:           fir.do_loop {{.*}} {
! Verify that `j` is allocated whithin the same scope of its block (i.e. inside
! the `task` loop).
! CHECK:             fir.alloca i32 {bindc_name = "j", {{.*}}}
! CHECK:           }
! CHECK:         }
! CHECK:       }
