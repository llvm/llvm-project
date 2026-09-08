! Each sub-file exercises a different unstructured-CFG pattern inside an
! `acc loop` whose default parallelism resolves to `independent`.

! RUN: split-file %s %t

! By default (--emit-independent-loops-as-unstructured=true), the loops are
! lowered to `acc.loop` operations.
! RUN: bbc -fopenacc -emit-hlfir %t/goto_one_level.f90 -o - | FileCheck %s --check-prefix=GOTO1-OK
! RUN: bbc -fopenacc -emit-hlfir %t/goto_with_intermediate.f90 -o - | FileCheck %s --check-prefix=GOTO2-OK
! RUN: bbc -fopenacc -emit-hlfir %t/collapse_cycle.f90 -o - | FileCheck %s --check-prefix=CCYCLE-OK
! RUN: bbc -fopenacc -emit-hlfir %t/cache_exit.f90 -o - | FileCheck %s --check-prefix=CEXIT-OK
! RUN: bbc -fopenacc -emit-hlfir %t/cache_select_case.f90 -o - | FileCheck %s --check-prefix=CCASE-OK

! With --emit-independent-loops-as-unstructured=false, the TODO is emitted.
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/goto_one_level.f90 -o - 2>&1 | FileCheck %s --check-prefix=GOTO1
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/goto_with_intermediate.f90 -o - 2>&1 | FileCheck %s --check-prefix=GOTO2
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/collapse_cycle.f90 -o - 2>&1 | FileCheck %s --check-prefix=CCYCLE
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/cache_exit.f90 -o - 2>&1 | FileCheck %s --check-prefix=CEXIT
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %t/cache_select_case.f90 -o - 2>&1 | FileCheck %s --check-prefix=CCASE

!--- goto_one_level.f90

! GOTO exits the inner `acc loop seq` (one level), landing in the body of
! the outer `acc loop gang vector`. Outer loop defaults to `independent`.
subroutine test_unstructured6(N, A, B)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
20  B(i) = A(i)
100 continue
end subroutine

! GOTO1: not yet implemented: unstructured do loop in independent OpenACC loop construct

! GOTO1-OK-LABEL: func.func @_QPtest_unstructured6
! GOTO1-OK: acc.loop {{.*}}gang vector
! GOTO1-OK: acc.loop

!--- goto_with_intermediate.f90

! Same as above but with intermediate code between the inner loop end and
! the GOTO target, exercising the jump-table dispatch path.
subroutine test_unstructured7(A, B, C, N)
  implicit real*8 (a-h, o-z)
  !$acc routine gang
  dimension A(*), B(*), C(*)
  !$acc loop gang vector
  do 100 i = 1, N
  !$acc loop seq
    do 10 j = 1, 1000
      if (A(i) .gt. B(i)) goto 20
10  continue
    C(i) = 999.0
20  B(i) = A(i)
100 continue
end subroutine

! GOTO2: not yet implemented: unstructured do loop in independent OpenACC loop construct

! GOTO2-OK-LABEL: func.func @_QPtest_unstructured7
! GOTO2-OK: acc.loop {{.*}}gang vector
! GOTO2-OK: acc.loop

!--- collapse_cycle.f90

! Orphan `acc loop collapse(2)` with an early-exit (CYCLE) - defaults to
! `independent` inside the (non-seq) acc routine.
subroutine test_unstructured_collapse_loop_only(a)
  !$acc routine gang
  integer :: i, j, jdiag
  real(8) :: a(:,:)
  jdiag = 4
  !$acc loop collapse(2)
  do j = 1, 8
    do i = 1, 8
      if (i == jdiag) then
        a(i, j) = 0.0d0
        cycle
      end if
      a(i, j) = real(i + j, 8)
    end do
  end do
end subroutine

! CCYCLE: not yet implemented: unstructured do loop in independent OpenACC loop construct

! CCYCLE-OK-LABEL: func.func @_QPtest_unstructured_collapse_loop_only
! CCYCLE-OK: acc.loop

!--- cache_exit.f90

! `acc loop` with `cache` directive and EXIT inside the body - the EXIT
! makes the loop unstructured. Orphan loop inside a (non-seq) acc routine
! defaults to `independent`.
subroutine test_cache_single_element()
  !$acc routine gang
  integer, parameter :: n = 10
  real, dimension(n) :: a, b
  integer :: i

  !$acc loop
  do i = 1, n
    !$acc cache(b(i))
    a(i) = b(i)
    if (a(i) > 100.0) exit
  end do
end subroutine

! CEXIT: not yet implemented: unstructured do loop in independent OpenACC loop construct

! CEXIT-OK-LABEL: func.func @_QPtest_cache_single_element
! CEXIT-OK: acc.loop

!--- cache_select_case.f90

! `acc loop` with `cache` directive and SELECT CASE inside the body - the
! SELECT CASE makes the loop's body have unstructured CFG. Orphan loop
! inside a (non-seq) acc routine defaults to `independent`.
subroutine test_cache_nonunit_lb()
  !$acc routine gang
  integer :: arr(10:20)
  integer :: i

  !$acc loop
  do i = 10, 20
    !$acc cache(arr(15))
    select case (mod(i, 3))
    case (0)
      arr(i) = i * 2
    case (1)
      arr(i) = i * 3
    case default
      arr(i) = i
    end select
  end do
end subroutine

! CCASE: not yet implemented: unstructured do loop in independent OpenACC loop construct

! CCASE-OK-LABEL: func.func @_QPtest_cache_nonunit_lb
! CCASE-OK: acc.loop
