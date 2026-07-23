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
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("x") -> !fir.ref<i32>
! CHECK-NOT: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("x") -> !fir.ref<i32>

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
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("x") -> !fir.ref<i32>
! CHECK-NOT: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("x") -> !fir.ref<i32>

! -----------------------------------------------------------------------
! private(x) private(x) -- duplicate across two separate clauses

subroutine test_private_two_clauses(i)
  integer :: x, i
  !$acc parallel loop private(x) private(x)
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_two_clauses
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("x") -> !fir.ref<i32>
! CHECK-NOT: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("x") -> !fir.ref<i32>

! -----------------------------------------------------------------------
! firstprivate(x, x)

subroutine test_firstprivate_pair(i)
  integer :: x, i
  !$acc parallel loop firstprivate(x, x)
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_firstprivate_pair
! CHECK: acc.firstprivate varPtr({{.*}}) recipe(@firstprivatization_ref_i32) name("x") -> !fir.ref<i32>
! CHECK-NOT: acc.firstprivate varPtr({{.*}}) recipe(@firstprivatization_ref_i32) name("x") -> !fir.ref<i32>

! -----------------------------------------------------------------------
! private(arr(1:5), arr(3)) -- contained array element in the same
! data-sharing kind

subroutine test_private_contained_array_parent_first(i)
  real :: arr(10)
  integer :: i
  !$acc parallel loop private(arr(1:5), arr(3))
  do i = 1, 10
    arr(i) = real(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_contained_array_parent_first
! arr(1:5) is privatized, and the contained arr(3) occurrence is removed before lowering.
! CHECK-NOT: acc.private {{.*}} name("arr(3)") ->
! CHECK: %[[ARR_PRIV:.*]] = acc.private {{.*}} name("arr(1:5)") ->
! CHECK-NOT: acc.private {{.*}} name("arr(3)") ->
! CHECK: acc.loop {{.*}}private(%[[ARR_PRIV]],

! -----------------------------------------------------------------------
! private(arr(3), arr(1:5)) -- contained array element appears first

subroutine test_private_contained_array_child_first(i)
  real :: arr(10)
  integer :: i
  !$acc parallel loop private(arr(3), arr(1:5))
  do i = 1, 10
    arr(i) = real(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_contained_array_child_first
! CHECK-NOT: acc.private {{.*}} name("arr(3)") ->
! CHECK: %[[ARR_PRIV:.*]] = acc.private {{.*}} name("arr(1:5)") ->
! CHECK-NOT: acc.private {{.*}} name("arr(3)") ->
! CHECK: acc.loop {{.*}}private(%[[ARR_PRIV]],

! -----------------------------------------------------------------------
! private(n%pt, n%pt%x) -- contained path in the same data-sharing kind

subroutine test_private_contained_component_parent_first(i)
  type point_t
    real :: x
    real :: y
  end type
  type nested_t
    type(point_t) :: pt
    integer :: tag
  end type
  type(nested_t) :: n
  integer :: i
  !$acc parallel loop private(n%pt, n%pt%x)
  do i = 1, 10
    n%pt%x = real(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_contained_component_parent_first
! n%pt is privatized, and the contained n%pt%x occurrence is removed before lowering.
! CHECK-NOT: acc.private {{.*}} name("n%pt%x") ->
! CHECK: %[[N_PT_PRIV:.*]] = acc.private {{.*}} name("n%pt") ->
! CHECK-NOT: acc.private {{.*}} name("n%pt%x") ->
! CHECK: acc.loop {{.*}}private(%[[N_PT_PRIV]],

! -----------------------------------------------------------------------
! private(n%pt%x, n%pt) -- contained path appears before the containing path

subroutine test_private_contained_component_child_first(i)
  type point_t
    real :: x
    real :: y
  end type
  type nested_t
    type(point_t) :: pt
    integer :: tag
  end type
  type(nested_t) :: n
  integer :: i
  !$acc parallel loop private(n%pt%x, n%pt)
  do i = 1, 10
    n%pt%x = real(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_contained_component_child_first
! CHECK-NOT: acc.private {{.*}} name("n%pt%x") ->
! CHECK: %[[N_PT_PRIV:.*]] = acc.private {{.*}} name("n%pt") ->
! CHECK-NOT: acc.private {{.*}} name("n%pt%x") ->
! CHECK: acc.loop {{.*}}private(%[[N_PT_PRIV]],
