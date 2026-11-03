! An `acc loop` whose default parallelism resolves to `independent` and whose
! body branching is not confined to it.
!
! The EXIT leaves the loop, so the loop stays unstructured and cannot be
! lowered with its bounds on the op. The patterns that do keep their branching
! inside the body live in ../acc-unstructured-loop-construct.f90.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s --check-prefix=CEXIT-OK
! RUN: %not_todo_cmd bbc -fopenacc -emit-hlfir --emit-independent-loops-as-unstructured=false %s -o - 2>&1 | FileCheck %s --check-prefix=CEXIT

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
