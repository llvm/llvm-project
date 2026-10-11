! Test that `lastprivate(conditional:)` instruments OpenMP atomic
! write/update/capture of the list item with the same canonical-index guarded
! commit used for ordinary assignments.  An atomic on a per-thread-private item
! is redundant but legal, and clang tracks atomic assignments to a conditional
! lastprivate item too, so the value must still be captured.

! RUN: bbc -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s
! RUN: %flang_fc1 -fopenmp -fopenmp-version=50 -emit-hlfir %s -o - | FileCheck %s

subroutine test_atomic_write(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) then
      !$omp atomic write
      x = i
    end if
  end do
end subroutine
! CHECK-LABEL: func.func @_QPtest_atomic_write
! CHECK: omp.atomic.write %[[XW:.*]] =
! CHECK: %[[VW:.*]] = fir.load %[[XW]]
! CHECK: %[[SXW:.*]] = fir.coordinate_of %[[STW:.*]], x
! CHECK: %[[SIW:.*]] = fir.coordinate_of %[[STW]], $x
! CHECK: %[[CURW:.*]] = fir.load %[[SIW]]
! CHECK: %[[CMPW:.*]] = arith.cmpi sge, %{{.*}}, %[[CURW]]
! CHECK: fir.if %[[CMPW]] {
! CHECK:   fir.store %[[VW]] to %[[SXW]]
! CHECK:   fir.store %{{.*}} to %[[SIW]]
! CHECK: }

subroutine test_atomic_update(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) then
      !$omp atomic update
      x = x + i
    end if
  end do
end subroutine
! CHECK-LABEL: func.func @_QPtest_atomic_update
! CHECK: omp.atomic.update %[[XU:.*]] :
! CHECK: %[[VU:.*]] = fir.load %[[XU]]
! CHECK: %[[SXU:.*]] = fir.coordinate_of %[[STU:.*]], x
! CHECK: %[[SIU:.*]] = fir.coordinate_of %[[STU]], $x
! CHECK: %[[CURU:.*]] = fir.load %[[SIU]]
! CHECK: %[[CMPU:.*]] = arith.cmpi sge, %{{.*}}, %[[CURU]]
! CHECK: fir.if %[[CMPU]] {
! CHECK:   fir.store %[[VU]] to %[[SXU]]
! CHECK:   fir.store %{{.*}} to %[[SIU]]
! CHECK: }

subroutine test_atomic_capture(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, v
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) then
      !$omp atomic capture
      v = x
      x = i
      !$omp end atomic
    end if
  end do
end subroutine
! The commit is injected after the whole omp.atomic.capture op, not inside it.
! CHECK-LABEL: func.func @_QPtest_atomic_capture
! CHECK: omp.atomic.capture {
! CHECK-NOT: arith.cmpi sge
! CHECK: }
! CHECK: %[[VC:.*]] = fir.load %[[XC:.*]]
! CHECK: %[[SXC:.*]] = fir.coordinate_of %[[STC:.*]], x
! CHECK: %[[SIC:.*]] = fir.coordinate_of %[[STC]], $x
! CHECK: %[[CURC:.*]] = fir.load %[[SIC]]
! CHECK: %[[CMPC:.*]] = arith.cmpi sge, %{{.*}}, %[[CURC]]
! CHECK: fir.if %[[CMPC]] {
! CHECK:   fir.store %[[VC]] to %[[SXC]]
! CHECK:   fir.store %{{.*}} to %[[SIC]]
! CHECK: }

subroutine test_atomic_capture_update(n, x)
  implicit none
  integer, intent(in) :: n
  integer, intent(inout) :: x
  integer :: i, v
  !$omp parallel do lastprivate(conditional: x)
  do i = 1, n
    if (mod(i, 2) == 0) then
      !$omp atomic capture
      v = x
      x = x + i
      !$omp end atomic
    end if
  end do
end subroutine
! A capture whose nested op is an atomic update is also committed after the
! whole omp.atomic.capture op.
! CHECK-LABEL: func.func @_QPtest_atomic_capture_update
! CHECK: omp.atomic.capture {
! CHECK:   omp.atomic.update
! CHECK: atomic_control
! CHECK-NOT: arith.cmpi sge
! CHECK: }
! CHECK: %[[VCU:.*]] = fir.load %[[XCU:.*]]
! CHECK: %[[SXCU:.*]] = fir.coordinate_of %[[STCU:.*]], x
! CHECK: %[[SICU:.*]] = fir.coordinate_of %[[STCU]], $x
! CHECK: %[[CURCU:.*]] = fir.load %[[SICU]]
! CHECK: %[[CMPCU:.*]] = arith.cmpi sge, %{{.*}}, %[[CURCU]]
! CHECK: fir.if %[[CMPCU]] {
! CHECK:   fir.store %[[VCU]] to %[[SXCU]]
! CHECK:   fir.store %{{.*}} to %[[SICU]]
! CHECK: }

subroutine test_atomic_capture_dest(n)
  implicit none
  integer, intent(in) :: n
  integer :: i, x, v
  x = 0
  !$omp parallel do lastprivate(conditional: v)
  do i = 1, n
    if (mod(i, 2) == 0) then
      !$omp atomic capture
      v = x
      x = x + i
      !$omp end atomic
    end if
  end do
end subroutine
! When the conditional-LP item is the capture READ destination (v), the value
! captured into it must also be committed after the whole omp.atomic.capture op.
! CHECK-LABEL: func.func @_QPtest_atomic_capture_dest
! CHECK: omp.atomic.capture {
! CHECK:   omp.atomic.read
! CHECK: atomic_control
! CHECK-NOT: arith.cmpi sge
! CHECK: }
! CHECK: %[[VD:.*]] = fir.load %[[XD:.*]]
! CHECK: %[[SVD:.*]] = fir.coordinate_of %[[STD:.*]], v
! CHECK: %[[SID:.*]] = fir.coordinate_of %[[STD]], $v
! CHECK: %[[CURD:.*]] = fir.load %[[SID]]
! CHECK: %[[CMPD:.*]] = arith.cmpi sge, %{{.*}}, %[[CURD]]
! CHECK: fir.if %[[CMPD]] {
! CHECK:   fir.store %[[VD]] to %[[SVD]]
! CHECK:   fir.store %{{.*}} to %[[SID]]
! CHECK: }
