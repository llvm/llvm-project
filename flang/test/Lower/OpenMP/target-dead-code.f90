! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | FileCheck %s --check-prefix=FIR

! Test that OpenMP target regions in dead code are marked for elimination

! Test 1: if (.false.) with target - should be marked unreachable
! FIR-LABEL: func.func @_QPtest_dead_simple
! FIR: %[[FALSE:.*]] = arith.constant false
! FIR: fir.if %[[FALSE]] {
! FIR: omp.target
! FIR: } {omp.target_unreachable}
subroutine test_dead_simple()
  real :: v
  if (.false.) then
    !$omp target map(tofrom:v)
    v = 1.0
    !$omp end target
  end if
end subroutine

! Test 2: Live target - should NOT be marked
! FIR-LABEL: func.func @_QPtest_live_simple
! FIR: omp.target
! FIR-NOT: omp.target_unreachable
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
  ! Dead - should be marked
  ! FIR: fir.if %{{.*}} {
  if (.false.) then
    !$omp target map(tofrom:v)
    ! FIR: omp.target
    ! FIR: } {omp.target_unreachable}
    v = 3.0
    !$omp end target
  end if
  ! Live - should NOT be marked
  !$omp target map(tofrom:v)
  ! FIR: omp.target
  ! FIR-NOT: omp.target_unreachable
  v = 4.0
  !$omp end target
end subroutine

! Test 4: Nested - outer false
! FIR-LABEL: func.func @_QPtest_nested_outer_false
subroutine test_nested_outer_false()
  real :: v
  ! FIR: fir.if %{{.*}} {
  if (.false.) then
    ! FIR: fir.if %{{.*}} {
    if (.true.) then
      ! FIR: omp.target
      ! FIR: } {omp.target_unreachable}
      !$omp target map(tofrom:v)
      v = 5.0
      !$omp end target
    end if
  end if
end subroutine

! Test 5: Parameter constant
! FIR-LABEL: func.func @_QPtest_parameter
subroutine test_parameter()
  real :: v
  logical, parameter :: DEAD = .false.
  ! FIR: fir.if %{{.*}} {
  if (DEAD) then
    ! FIR: omp.target
    ! FIR: } {omp.target_unreachable}
    !$omp target map(tofrom:v)
    v = 6.0
    !$omp end target
  end if
end subroutine
