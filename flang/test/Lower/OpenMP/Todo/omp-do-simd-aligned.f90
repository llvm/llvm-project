! This test checks lowering of OpenMP do simd aligned() pragma

! RUN: %not_todo_cmd bbc -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s
subroutine testDoSimdAligned(int_array)
        use iso_c_binding
        type(c_ptr) :: int_array
!CHECK: not yet implemented: Unhandled clause ALIGNED in DO SIMD construct
!$omp do simd aligned(int_array)
        do index_ = 1, 10
          call c_test_call(int_array)
        end do
!$omp end do simd

end subroutine testDoSimdAligned

