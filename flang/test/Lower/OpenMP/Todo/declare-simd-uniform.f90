! RUN: %not_todo_cmd %flang_fc1 -emit-fir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: Unhandled clause UNIFORM in DECLARE SIMD construct
subroutine declare_simd_uniform(x, y)
    real(8), pointer, intent(inout) :: x(:)
    real(8), pointer, intent(in)    :: y(:)
    !$omp declare simd uniform(x, y)
end subroutine declare_simd_uniform

