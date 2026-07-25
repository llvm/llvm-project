! A user-defined reduction on an allocatable (or pointer) variable of a trivial
! by-value type (integer/real) is not yet lowered: the directive materializes a
! by-value op (e.g. @..._i32) from the declared element type, but the reduction
! operand is a boxed, by-ref allocatable, so the clause looks up a by-ref name
! (@..._byref_i32) the directive never emitted and reaches a clean TODO rather
! than binding a by-value op to a boxed operand (which produced invalid IR
! before). Creating the boxed-type op for such operands is tracked by
! https://github.com/llvm/llvm-project/pull/186765. Character and derived-type
! allocatable reductions (by-ref op and by-ref operand) do lower and are covered
! by declare-reduction-character-allocatable.f90.

! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -o - %s 2>&1 | FileCheck %s

! CHECK: not yet implemented: OpenMP user-defined named reduction is not yet lowered for this variable's shape

program p
  integer, allocatable :: a
  real, allocatable :: b
  integer :: i
  !$omp declare reduction(myadd: integer, real : omp_out = omp_out + omp_in) &
  !$omp   initializer(omp_priv = 0)
  allocate(a); a = 0
  allocate(b); b = 0.0
  !$omp parallel do reduction(myadd: a)
  do i = 1, 5
    a = a + i
  end do
  !$omp parallel do reduction(myadd: b)
  do i = 1, 5
    b = b + real(i)
  end do
  print *, a, b
end program
