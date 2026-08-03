! Loop-associated metadirective variants cannot yet reconstruct variant-local
! data-sharing relationships. Cover selected data environments, explicit
! data-sharing clauses, enclosing data environments, and eager privatization.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/parallel-do.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=DATA-ENV %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/private.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=DATA-SHARING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -mmlir --enable-delayed-privatization=false \
! RUN:   -o - %t/eager.f90 2>&1 | FileCheck --check-prefix=EAGER %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/enclosing-static.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=ENCLOSING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/enclosing-dynamic.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=ENCLOSING %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/selected-metadirective.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=SELECTED %s

! DATA-ENV: not yet implemented: data-environment construct in loop-associated METADIRECTIVE variant
! DATA-SHARING: not yet implemented: data-sharing clause in loop-associated METADIRECTIVE variant
! EAGER: not yet implemented: loop-associated METADIRECTIVE with eager privatization
! ENCLOSING: not yet implemented: loop-associated METADIRECTIVE nested in an OpenMP data environment
! SELECTED: not yet implemented: data-environment construct in METADIRECTIVE variant

!--- parallel-do.f90
subroutine test_parallel_do(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: parallel do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- private.f90
subroutine test_private(n, a)
  integer :: n, a(n), i, x
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do private(x)) &
  !$omp & otherwise(nothing)
  do i = 1, n
    x = i
    a(i) = x
  end do
end subroutine

!--- eager.f90
subroutine test_eager_privatization(n, a)
  integer :: n, a(n), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- enclosing-static.f90
subroutine test_static_do_in_parallel(n, a, after)
  integer :: n, a(n), after, i
  i = 0
  !$omp parallel num_threads(1) shared(n, a, after, i)
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end parallel
end subroutine

!--- enclosing-dynamic.f90
subroutine test_do_in_parallel(flag, n, a, after)
  logical, intent(in) :: flag
  integer :: n, a(n), after, i
  i = 0
  !$omp parallel num_threads(1) shared(flag, n, a, after, i)
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end parallel
end subroutine

!--- selected-metadirective.f90
subroutine test_do_in_selected_parallel(flag, n, a, after)
  logical, intent(in) :: flag
  integer :: n, a(n), after, i
  i = 0
  !$omp begin metadirective &
  !$omp & when(implementation={vendor(llvm)}: &
  !$omp &   parallel num_threads(1) shared(flag, n, a, after, i)) &
  !$omp & otherwise(nothing)
  !$omp metadirective &
  !$omp & when(user={condition(flag)}: do) &
  !$omp & otherwise(nothing)
  do i = 1, n
    a(i) = i
  end do
  after = i
  !$omp end metadirective
end subroutine
