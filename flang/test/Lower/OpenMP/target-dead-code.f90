! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=FIR

! Test that OpenMP target regions in dead code are deleted

! Test 1: if (.false.) with target - target should be deleted
! FIR-LABEL: func.func @_QPtest_dead_simple
! FIR: %[[FALSE:.*]] = arith.constant false
! FIR: fir.if %[[FALSE]] {
! FIR-NOT: omp.target
subroutine test_dead_simple()
  real :: v
  if (.false.) then
    !$omp target map(tofrom:v)
    v = 1.0
    !$omp end target
  end if
end subroutine

! Test 2: Live target - should remain
! FIR-LABEL: func.func @_QPtest_live_simple
! FIR: omp.target
subroutine test_live_simple()
  real :: v
  !$omp target map(tofrom:v)
  v = 2.0
  !$omp end target
end subroutine

! Test 3: Mixed dead and live
! FIR-LABEL: func.func @_QPtest_mixed
subroutine test_mixed()
  real :: v
  ! Dead - should be deleted
  ! FIR: fir.if %{{.*}} {
  if (.false.) then
    !$omp target map(tofrom:v)
    v = 3.0
    !$omp end target
  end if
  ! FIR-NOT: omp.target
  ! Live - should remain (expect exactly 1 omp.target in function)
  !$omp target map(tofrom:v)
  ! FIR: omp.target
  v = 4.0
  !$omp end target
end subroutine

! Test 4: Nested - outer false, target should be deleted
! FIR-LABEL: func.func @_QPtest_nested_outer_false
subroutine test_nested_outer_false()
  real :: v
  ! FIR: fir.if %{{.*}} {
  if (.false.) then
    if (.true.) then
      !$omp target map(tofrom:v)
      v = 5.0
      !$omp end target
    end if
  end if
  ! FIR-NOT: omp.target
end subroutine

! Test 5: Parameter constant - target should be deleted
! FIR-LABEL: func.func @_QPtest_parameter
subroutine test_parameter()
  real :: v
  logical, parameter :: DEAD = .false.
  ! FIR: fir.if %{{.*}} {
  if (DEAD) then
    !$omp target map(tofrom:v)
    v = 6.0
    !$omp end target
  end if
  ! FIR-NOT: omp.target
end subroutine

! FIR-LABEL: func.func @_QPtest_outer
subroutine test_outer
  implicit none
contains
  subroutine unused_sub()
    real :: v
    !$omp target map(tofrom: v)
      v = 5.0
    !$omp end target
  end subroutine
  ! FIR-NOT: omp.target
end subroutine
