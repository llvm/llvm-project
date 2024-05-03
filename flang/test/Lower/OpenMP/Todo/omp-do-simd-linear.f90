! This test checks lowering of OpenMP do simd linear() pragma

! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
subroutine testDoSimdLinear(int_array)
        integer :: int_array(*)
!CHECK: not yet implemented: Unhandled clause LINEAR in DO SIMD construct
!$omp do simd linear(int_array)
        do index_ = 1, 10
        end do
!$omp end do simd

end subroutine testDoSimdLinear

