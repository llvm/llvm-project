! Test for DO SIMD with the same variable in both firstprivate and lastprivate clauses
! This tests the fix for issue #168306

! RUN: %flang_fc1 -fopenmp -mmlir --enable-delayed-privatization-staging=true -emit-hlfir %s -o - | FileCheck %s

! Test case 1: Basic test with firstprivate + lastprivate on same variable
! CHECK-LABEL: func.func @_QPdo_simd_first_last_same_var
subroutine do_simd_first_last_same_var()
  integer :: a
  integer :: i
  a = 10

  ! CHECK:      omp.wsloop
  ! CHECK-SAME: private(@{{.*}}firstprivate{{.*}} %{{.*}} -> %[[FIRSTPRIV_A:.*]], @{{.*}}private{{.*}} %{{.*}} -> %[[PRIV_I:.*]] : !fir.ref<i32>, !fir.ref<i32>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT: private
  ! CHECK-NEXT: omp.loop_nest (%[[IV:.*]]) : i32
  !$omp do simd firstprivate(a) lastprivate(a)
  do i = 1, 1
    ! CHECK: %[[FIRSTPRIV_A_DECL:.*]]:2 = hlfir.declare %[[FIRSTPRIV_A]]
    ! CHECK: %[[PRIV_I_DECL:.*]]:2 = hlfir.declare %[[PRIV_I]]
    ! The private copy should be initialized from firstprivate (value 10)
    ! and then modified to 20
    a = 20
  end do
  !$omp end do simd
  ! After the loop, 'a' should be 20 due to lastprivate
end subroutine do_simd_first_last_same_var

! Test case 2: Test with lastprivate and firstprivate in reverse order
! CHECK-LABEL: func.func @_QPdo_simd_last_first_reverse
subroutine do_simd_last_first_reverse()
  integer :: a
  integer :: i
  a = 10

  ! CHECK:      omp.wsloop
  ! CHECK-SAME: private(@{{.*}}firstprivate{{.*}} %{{.*}} -> %[[FIRSTPRIV_A:.*]], @{{.*}}private{{.*}} %{{.*}} -> %[[PRIV_I:.*]] : !fir.ref<i32>, !fir.ref<i32>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT: private
  !$omp do simd lastprivate(a) firstprivate(a)
  do i = 1, 1
    a = 20
  end do
  !$omp end do simd
end subroutine do_simd_last_first_reverse

! Test case 3: Multiple variables with mixed privatization
! CHECK-LABEL: func.func @_QPdo_simd_multiple_vars
subroutine do_simd_multiple_vars()
  integer :: a, b, c
  integer :: i
  a = 10
  b = 20
  c = 30

  ! CHECK:      omp.wsloop
  ! CHECK-SAME: private(@{{.*}}firstprivate{{.*}} %{{.*}} -> %{{.*}}, @{{.*}}firstprivate{{.*}} %{{.*}} -> %{{.*}}, @{{.*}}private{{.*}} %{{.*}} -> %{{.*}} : !fir.ref<i32>, !fir.ref<i32>, !fir.ref<i32>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT: private
  !$omp do simd firstprivate(a, b) lastprivate(a) private(c)
  do i = 1, 5
    a = a + 1
    b = b + 1
    c = i
  end do
  !$omp end do simd
end subroutine do_simd_multiple_vars

! Test case 4: Reproducer from issue #168306
! CHECK-LABEL: func.func @_QPissue_168306_reproducer
subroutine issue_168306_reproducer()
  integer :: a
  integer :: i
  a = 10

  ! CHECK:      omp.wsloop
  ! CHECK-SAME: private(@{{.*}}firstprivate{{.*}} %{{.*}} -> %[[FIRSTPRIV_A:.*]], @{{.*}}private{{.*}} %{{.*}} -> %[[PRIV_I:.*]] : !fir.ref<i32>, !fir.ref<i32>)
  ! CHECK-NEXT: omp.simd
  ! CHECK-NOT: private
  !$omp do simd lastprivate(a) firstprivate(a)
  do i = 1, 1
    ! Inside the loop, 'a' should start at 10 (from firstprivate)
    ! This is the key behavior that was broken
    a = 20
  end do
  !$omp end do simd
  ! After the loop, 'a' should be 20 (from lastprivate)
end subroutine issue_168306_reproducer
