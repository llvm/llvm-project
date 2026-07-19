! RUN: bbc -fopenacc -emit-hlfir %s -o - 2>/dev/null | FileCheck %s

! Check that same-kind duplicate variables in OpenACC private/firstprivate
! clauses lower without failure, and that each variable produces exactly one
! acc.private / acc.firstprivate op (deduplication by rewrite-parse-tree).

! -----------------------------------------------------------------------
! private(x, x) -- duplicate within one clause

subroutine test_private_pair(i)
  integer :: x, i
  !$acc parallel loop private(x, x)
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_pair
! x is privatized exactly once.
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {name = "x"}
! CHECK-NOT: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {name = "x"}

! -----------------------------------------------------------------------
! private(x, x, x) -- two duplicates (from the triple-occurrence review note)

subroutine test_private_triple(i)
  integer :: x, i
  !$acc parallel loop private(x, x, x)
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_triple
! x is privatized exactly once even with three source occurrences.
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {name = "x"}
! CHECK-NOT: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {name = "x"}

! -----------------------------------------------------------------------
! private(x) private(x) -- duplicate across two separate clauses

subroutine test_private_two_clauses(i)
  integer :: x, i
  !$acc parallel loop private(x) private(x)
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_two_clauses
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {name = "x"}
! CHECK-NOT: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) -> !fir.ref<i32> {name = "x"}

! -----------------------------------------------------------------------
! firstprivate(x, x)

subroutine test_firstprivate_pair(i)
  integer :: x, i
  !$acc parallel loop firstprivate(x, x)
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_firstprivate_pair
! CHECK: acc.firstprivate varPtr({{.*}}) recipe(@firstprivatization_ref_i32) -> !fir.ref<i32> {name = "x"}
! CHECK-NOT: acc.firstprivate varPtr({{.*}}) recipe(@firstprivatization_ref_i32) -> !fir.ref<i32> {name = "x"}
