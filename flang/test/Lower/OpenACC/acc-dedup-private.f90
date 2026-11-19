! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

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
! private(x, x, x) -- two duplicates

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
! A contained private object is dropped in favor of its containing object.

subroutine test_private_contained_after(i)
  real :: a(10)
  integer :: i
  !$acc parallel loop private(a) private(a(1:5))
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_contained_after
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_10xf32) name("a") -> !fir.ref<!fir.array<10xf32>>
! CHECK-NOT: name("a(1:5)")

! -----------------------------------------------------------------------
! Source order does not affect which object is retained.

subroutine test_private_contained_before(i)
  real :: a(10)
  integer :: i
  !$acc parallel loop private(a(1:5)) private(a)
  do i = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_contained_before
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_10xf32) name("a") -> !fir.ref<!fir.array<10xf32>>
! CHECK-NOT: name("a(1:5)")

! -----------------------------------------------------------------------
! A trailing whole array subsumes multiple unknown selectors before those
! selectors are diagnosed as distinct parts of the same array.

subroutine test_private_container_last(a, i, j, k)
  real :: a(10)
  integer :: i, j, k
  !$acc parallel loop private(a(i), a(j)) private(a)
  do k = 1, 10
  end do
end subroutine

! CHECK-LABEL: func.func @_QPtest_private_container_last
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_10xf32) name("a") -> !fir.ref<!fir.array<10xf32>>
! CHECK-NOT: name("a({{.*}})")

! -----------------------------------------------------------------------
! Standalone LOOP headers finalize their clauses before the loop body.

subroutine test_standalone_private_pair()
  integer :: x, i
  !$acc parallel
  !$acc loop private(x, x)
  do i = 1, 10
    x = i
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_standalone_private_pair
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_i32) name("x") -> !fir.ref<i32>
! CHECK-NOT: acc.private {{.*}} name("x")
! CHECK: acc.loop

subroutine test_standalone_private_container_last(a, i, j)
  real :: a(10)
  integer :: i, j, k
  !$acc parallel
  !$acc loop private(a(i), a(j)) private(a)
  do k = 1, 10
    a(k) = real(k)
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_standalone_private_container_last
! CHECK-NOT: acc.private {{.*}} name("a({{.*}})")
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_10xf32) name("a") -> !fir.ref<!fir.array<10xf32>>
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("a({{.*}})")
! CHECK: acc.loop

subroutine test_standalone_private_container_first(a, i, j)
  real :: a(10)
  integer :: i, j, k
  !$acc parallel
  !$acc loop private(a) private(a(i), a(j))
  do k = 1, 10
    a(k) = real(k)
  end do
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPtest_standalone_private_container_first
! CHECK-NOT: acc.private {{.*}} name("a({{.*}})")
! CHECK: acc.private varPtr({{.*}}) recipe(@privatization_ref_10xf32) name("a") -> !fir.ref<!fir.array<10xf32>>
! CHECK-NOT: acc.private {{.*}} name("a")
! CHECK-NOT: acc.private {{.*}} name("a({{.*}})")
! CHECK: acc.loop
