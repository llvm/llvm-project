! Part 2 supports ordinary DO, SIMD, and DO SIMD loop replacement arms. Keep
! unsupported association mixes, directives, body shapes, and target
! host-evaluation paths diagnosed until their lowering is implemented.

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
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/empty-delimited-body.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=EMPTY %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/trailing-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TRAILING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/trailing-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TRAILING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/trailing-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TRAILING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/trailing-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=TRAILING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/barrier-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/barrier-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/barrier-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/barrier-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/barrier-nested-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/barrier-nested-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/barrier-nested-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/barrier-nested-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/barrier-nested-body-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/barrier-nested-body-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 \
! RUN:   -cpp -o - %t/barrier-nested-body-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -cpp -DOMP_52 -o - %t/barrier-nested-body-runtime.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=NESTING %s

! MIXED: not yet implemented: METADIRECTIVE with both block- and loop-associated variants
! DIRECTIVE: not yet implemented: loop-associated METADIRECTIVE variant other than DO, SIMD, or DO SIMD
! TARGET: not yet implemented: TARGET construct selected by METADIRECTIVE (host-eval)
! EMPTY: not yet implemented: loop-associated METADIRECTIVE without associated
! EMPTY-SAME: DO
! TRAILING: not yet implemented: loop-associated METADIRECTIVE with content following the associated DO
! NESTING: not yet implemented: BARRIER closely nested in loop-associated METADIRECTIVE variant

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

!--- empty-delimited-body.f90
subroutine test_empty_delimited_body(n, a)
  integer :: n, a(n), i
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  !$omp end metadirective
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- trailing-static.f90
subroutine test_trailing_static(n, a, x)
  integer :: n, a(n), x, i
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    a(i) = i
  end do
  x = 42
  !$omp end metadirective
end subroutine

!--- trailing-runtime.f90
subroutine test_trailing_runtime(flag, n, a, x)
  logical, intent(in) :: flag
  integer :: n, a(n), x, i
  !$omp begin metadirective &
  !$omp & when(user={condition(flag)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    a(i) = i
  end do
  x = 42
  !$omp end metadirective
end subroutine

!--- barrier-static.f90
subroutine test_barrier_static(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    !$omp barrier
    a(i) = i
  end do
end subroutine

!--- barrier-runtime.f90
subroutine test_barrier_runtime(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    !$omp barrier
    a(i) = i
  end do
end subroutine

!--- barrier-nested-static.f90
subroutine test_barrier_nested_static(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    !$omp metadirective &
    !$omp & when(implementation={vendor(llvm)}: barrier) &
#ifdef OMP_52
    !$omp & otherwise(nothing)
#else
    !$omp & default(nothing)
#endif
    a(i) = i
  end do
end subroutine

!--- barrier-nested-runtime.f90
subroutine test_barrier_nested_runtime(flag, n, a)
  logical, intent(in) :: flag
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    !$omp metadirective &
    !$omp & when(user={condition(flag)}: barrier) &
#ifdef OMP_52
    !$omp & otherwise(nothing)
#else
    !$omp & default(nothing)
#endif
    a(i) = i
  end do
end subroutine

!--- barrier-nested-body-static.f90
subroutine test_barrier_nested_body_static(n)
  integer :: n, i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    !$omp begin metadirective &
    !$omp & when(implementation={vendor(llvm)}: nothing) &
#ifdef OMP_52
    !$omp & otherwise(parallel)
#else
    !$omp & default(parallel)
#endif
    !$omp barrier
    !$omp end metadirective
  end do
end subroutine

!--- barrier-nested-body-runtime.f90
subroutine test_barrier_nested_body_runtime(flag, n)
  logical, intent(in) :: flag
  integer :: n, i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
#ifdef OMP_52
  !$omp & otherwise(nothing)
#else
  !$omp & default(nothing)
#endif
  do i = 1, n
    !$omp begin metadirective &
    !$omp & when(user={condition(flag)}: parallel) &
#ifdef OMP_52
    !$omp & otherwise(nothing)
#else
    !$omp & default(nothing)
#endif
    !$omp barrier
    !$omp end metadirective
  end do
end subroutine
