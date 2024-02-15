! This test checks lowering of OpenMP do simd simdlen() pragma

! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
subroutine testDoSimdSimdlen(int_array)
        integer :: int_array(*)
!CHECK: not yet implemented: Unhandled clause SIMDLEN in DO SIMD construct
!$omp do simd simdlen(4)
        do index_ = 1, 10
        end do
!$omp end do simd

end subroutine testDoSimdSimdlen

