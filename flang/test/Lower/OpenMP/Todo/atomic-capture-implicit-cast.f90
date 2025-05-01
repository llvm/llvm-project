!RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

!CHECK: not yet implemented: atomic capture requiring implicit type casts 
subroutine capture_with_convert_f32_to_i32()
    implicit none
    integer :: k, v, i

    k = 1
    v = 0

    !$omp atomic capture
    v = k
    k = (i + 1) * 3.14
    !$omp end atomic
end subroutine

subroutine capture_with_convert_i32_to_f64()
    real(8) :: x
    integer :: v
    x = 1.0
    v = 0
    !$omp atomic capture
    v = x
    x = v
    !$omp end atomic
end subroutine capture_with_convert_i32_to_f64

subroutine capture_with_convert_f64_to_i32()
    integer :: x
    real(8) :: v
    x = 1
    v = 0
    !$omp atomic capture
    x = v
    v = x
    !$omp end atomic
end subroutine capture_with_convert_f64_to_i32

subroutine capture_with_convert_i32_to_f32()
    real(4) :: x
    integer :: v
    x = 1.0
    v = 0
    !$omp atomic capture
    v = x
    x = x + v
    !$omp end atomic
end subroutine capture_with_convert_i32_to_f32
