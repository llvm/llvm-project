! Test diagnostics for unsupported statements between a loop-associated
! metadirective and its associated DO.

! RUN: split-file %s %t
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/threadprivate.f90 2>&1 | FileCheck --check-prefix=TP %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/requires.f90 2>&1 | FileCheck --check-prefix=REQ %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/prefetch.f90 2>&1 | FileCheck --check-prefix=PREFETCH %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/begin-prefetch.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=BEGIN-PREFETCH %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/entry.f90 2>&1 | FileCheck --check-prefix=ENTRY %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -o - %t/format.f90 2>&1 | FileCheck --check-prefix=FORMAT %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -fopenacc -o - %t/acc-declare.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=ACC %s
! RUN: %not_todo_cmd %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=52 \
! RUN:   -fopenacc -o - %t/acc-routine.f90 2>&1 \
! RUN:   | FileCheck --check-prefix=ROUTINE %s

! TP: not yet implemented: THREADPRIVATE directive between loop-associated
! TP-SAME: METADIRECTIVE and its associated DO
! REQ: not yet implemented: REQUIRES directive between loop-associated
! REQ-SAME: METADIRECTIVE and its associated DO
! PREFETCH: not yet implemented: PREFETCH compiler directive between
! PREFETCH-SAME: loop-associated METADIRECTIVE and its associated DO
! BEGIN-PREFETCH: not yet implemented: PREFETCH compiler directive between
! BEGIN-PREFETCH-SAME: loop-associated METADIRECTIVE and its associated DO
! ENTRY: not yet implemented: ENTRY statement between loop-associated
! ENTRY-SAME: METADIRECTIVE and its associated DO
! FORMAT: not yet implemented: FORMAT statement between loop-associated
! FORMAT-SAME: METADIRECTIVE and its associated DO
! ACC: not yet implemented: OpenACC DECLARE directive between loop-associated
! ACC-SAME: METADIRECTIVE and its associated DO
! ROUTINE: not yet implemented: OpenACC ROUTINE directive between
! ROUTINE-SAME: loop-associated METADIRECTIVE and its associated DO

!--- threadprivate.f90
subroutine threadprivate_between(n, a)
  integer :: n, a(n), i
  real, save :: p
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  !$omp threadprivate(p)
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- requires.f90
subroutine requires_between()
  integer :: a(10), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  !$omp requires unified_address
  do i = 1, 10
    a(i) = i
  end do
end subroutine

!--- prefetch.f90
subroutine prefetch_between(a, n)
  integer :: a(n), n, i, idx
  logical :: choose
  external :: idx, choose
  !$omp metadirective &
  !$omp & when(user={condition(choose())}: do) &
  !$omp & otherwise(nothing)
  !dir$ prefetch a(idx())
  do i = 1, n
    a(i) = i
  end do
end subroutine

!--- begin-prefetch.f90
subroutine begin_prefetch_between(a, n)
  integer :: a(n), n, i, idx
  logical :: choose
  external :: idx, choose
  !$omp begin metadirective &
  !$omp & when(user={condition(choose())}: do) &
  !$omp & otherwise(nothing)
  !dir$ prefetch a(idx())
  do i = 1, n
    a(i) = i
  end do
  !$omp end metadirective
end subroutine

!--- entry.f90
subroutine entry_between()
  integer :: a(10), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  entry alternate_entry()
  do i = 1, 10
    a(i) = i
  end do
end subroutine

!--- acc-routine.f90
subroutine acc_routine_between(a)
  integer :: a(10), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  !$acc routine seq
  do i = 1, 10
    a(i) = i
  end do
end subroutine

!--- format.f90
subroutine format_between()
  integer :: a(10), i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
100 format(i0)
  do i = 1, 10
    a(i) = i
  end do
end subroutine

!--- acc-declare.f90
subroutine acc_declare_between()
  integer, save :: a(10)
  integer :: i
  !$omp metadirective &
  !$omp & when(implementation={vendor(llvm)}: do) &
  !$omp & otherwise(nothing)
  !$acc declare create(a)
  do i = 1, 10
    a(i) = i
  end do
end subroutine
