! Part 2 supports ordinary DO, SIMD, and DO SIMD loop replacement arms. Keep
! unsupported association mixes, directives, and target host-evaluation paths
! diagnosed until their lowering is implemented.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/mixed-association.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=MIXED %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/loop-directive.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=DIRECTIVE %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/target-loop.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TARGET %s

! MIXED: not yet implemented: METADIRECTIVE with both block- and loop-associated variants
! DIRECTIVE: not yet implemented: loop-associated METADIRECTIVE variant other than DO, SIMD, or DO SIMD
! TARGET: not yet implemented: TARGET construct selected by METADIRECTIVE (host-eval)

!--- mixed-association.f90
subroutine test_single_fallback(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(single)
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

!--- loop-directive.f90
subroutine test_loop(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: loop bind(thread)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- target-loop.f90
subroutine test_target_loop()
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: target teams distribute parallel do) &
  !$omp & otherwise(nothing)
  do i = 1, 100
  end do
end subroutine
