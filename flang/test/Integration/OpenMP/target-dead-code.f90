!===----------------------------------------------------------------------===!
! This directory can be used to add Integration tests involving multiple
! stages of the compiler (for eg. from Fortran to LLVM IR). It should not
! contain executable tests. We should only add tests here sparingly and only
! if there is no other way to test. Repeat this message in each test that is
! added to this directory and sub-directories.
!===----------------------------------------------------------------------===!

! RUN: %flang_fc1 -emit-hlfir -fopenmp %s -o - | tco -test-gen | FileCheck %s --check-prefixes=HOST,ALL
! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-is-target-device %s -o - | tco -test-gen | FileCheck %s --check-prefixes=DEVICE,ALL

! Test that OpenMP target regions in dead code are deleted from both host and
! target device.

! Test 1: if (.false.) with target - target should be deleted
! HOST-LABEL: llvm.func @_QPtest_dead_simple
! HOST-NOT: omp.target
! HOST: llvm.return
! DEVICE-NOT: llvm.func @_QPtest_dead_simple
subroutine test_dead_simple()
  real :: v
  if (.false.) then
    !$omp target map(tofrom:v)
    v = 1.0
    !$omp end target
  end if
end subroutine

! Test 2: Live target - should remain
! ALL-LABEL: llvm.func @_QPtest_live_simple
! ALL: omp.target
! ALL: llvm.return
subroutine test_live_simple()
  real :: v
  if (.true.) then
    !$omp target map(tofrom:v)
    v = 2.0
    !$omp end target
  end if
end subroutine

! Test 3: Mixed dead and live
! ALL-LABEL: llvm.func @_QPtest_mixed
subroutine test_mixed()
  real :: v
  ! Dead - should be deleted
  ! ALL-NOT: {{.*}} = llvm.mlir.constant(3.0{{.*}} : f32)
  if (.false.) then
    !$omp target map(tofrom:v)
    v = 3.0
    !$omp end target
  end if

  ! Live - should remain
  !$omp target map(tofrom:v)
  ! ALL: omp.target
  ! ALL: {{.*}} = llvm.mlir.constant(4.0{{.*}} : f32)
  v = 4.0
  !$omp end target

  ! Expect exactly 1 omp.target in function
  ! ALL-NOT: omp.target
  ! ALL: llvm.return
end subroutine

! Test 4: Nested - outer false, target should be deleted
! HOST-LABEL: llvm.func @_QPtest_nested_outer_false
! HOST-NOT: omp.target
! HOST: llvm.return
! DEVICE-NOT: llvm.func @_QPtest_nested_outer_false
subroutine test_nested_outer_false()
  real :: v
  if (.false.) then
    if (.true.) then
      !$omp target map(tofrom:v)
      v = 5.0
      !$omp end target
    end if
  end if
end subroutine

! Test 5: Nested - inner false, target should be deleted
! HOST-LABEL: llvm.func @_QPtest_nested_inner_false
! HOST-NOT: omp.target
! HOST: llvm.return
! DEVICE-NOT: llvm.func @_QPtest_nested_inner_false
subroutine test_nested_inner_false()
  real :: v
  if (.true.) then
    if (.false.) then
      !$omp target map(tofrom:v)
      v = 6.0
      !$omp end target
    end if
  end if
end subroutine

! Test 6: Nested - both true, target should remain
! ALL-LABEL: llvm.func @_QPtest_nested_both_true
! ALL: omp.target
! ALL: llvm.return
subroutine test_nested_both_true()
  real :: v
  if (.true.) then
    if (.true.) then
      !$omp target map(tofrom:v)
      v = 7.0
      !$omp end target
    end if
  end if
end subroutine

! Test 7: Multiple dead targets in dead branch - all should be deleted
! HOST-LABEL: llvm.func @_QPtest_multiple_dead_targets
! HOST-NOT: omp.target
! HOST: llvm.return
! DEVICE-NOT: llvm.func @_QPtest_multiple_dead_targets
subroutine test_multiple_dead_targets()
  real :: v
  if (.false.) then
    !$omp target map(tofrom:v)
    v = 8.0
    !$omp end target
    !$omp target map(tofrom:v)
    v = 9.0
    !$omp end target
    !$omp target map(tofrom:v)
    v = 10.0
    !$omp end target
  end if
end subroutine

! Test 8: Parameter constant - target should be deleted
! HOST-LABEL: llvm.func @_QPtest_parameter
! HOST-NOT: omp.target
! HOST: llvm.return
! DEVICE-NOT: llvm.func @_QPtest_parameter
subroutine test_parameter()
  real :: v
  logical, parameter :: DEAD = .false.
  if (DEAD) then
    !$omp target map(tofrom:v)
    v = 11.0
    !$omp end target
  end if
end subroutine

! Test 9: Unused nested subroutine - target should be deleted
! HOST-LABEL: llvm.func @_QPtest_outer
! HOST-NOT: omp.target
! HOST: llvm.return
! DEVICE-NOT: llvm.func @_QPtest_outer
subroutine test_outer
  implicit none
contains
  subroutine unused_sub()
    real :: v
    !$omp target map(tofrom: v)
      v = 12.0
    !$omp end target
  end subroutine
end subroutine

! Test 10: if (.false.) with else - then-branch target deleted, else-branch remains
! ALL-LABEL: llvm.func @_QPtest_if_else_false
subroutine test_if_else_false()
  real :: v
  ! Dead then-branch - target should be deleted
  ! ALL-NOT: {{.*}} = llvm.mlir.constant(1.3{{.*}}e+01 : f32)
  if (.false.) then
    !$omp target map(tofrom:v)
    v = 13.0
    !$omp end target
  else
    ! Live else-branch - target should remain
    !$omp target map(tofrom:v)
    ! ALL: omp.target
    ! ALL: {{.*}} = llvm.mlir.constant(1.4{{.*}}e+01 : f32)
    v = 14.0
    !$omp end target
  end if
  ! Expect exactly 1 omp.target in function
  ! ALL-NOT: omp.target
  ! ALL: llvm.return
end subroutine

! Test 11: Runtime condition - target should remain unchanged
! ALL-LABEL: llvm.func @_QPtest_runtime_condition
! ALL: omp.target
! ALL: llvm.return
subroutine test_runtime_condition(cond)
  logical, intent(in) :: cond
  real :: v
  if (cond) then
    !$omp target map(tofrom:v)
    v = 15.0
    !$omp end target
  end if
end subroutine

! Test 12: Target nested in unreachable block - target should be deleted
! ALL-LABEL: llvm.func @_QPtest_nested_in_unreachable_block
subroutine test_nested_in_unreachable_block()
  real :: v
  go to 10
  ! Unreachable block: even though condition is .true., the block itself is dead
  ! ALL-NOT: {{.*}} = llvm.mlir.constant(1.6{{.*}}e+01 : f32)
  if (.true.) then
    !$omp target map(tofrom:v)
    v = 16.0
    !$omp end target
  end if
10 continue
  ! Reachable - target should remain
  !$omp target map(tofrom:v)
  ! ALL: omp.target
  ! ALL: {{.*}} = llvm.mlir.constant(1.7{{.*}}e+01 : f32)
  v = 17.0
  !$omp end target
  ! Expect exactly 1 omp.target in function
  ! ALL-NOT: omp.target
  ! ALL: llvm.return
end subroutine

! Test 13: Multiple targets in unreachable blocks - all should be deleted
! ALL-LABEL: llvm.func @_QPtest_multiple_unreachable_blocks
subroutine test_multiple_unreachable_blocks()
  real :: v
  go to 30
  ! First unreachable block - target should be deleted
  ! ALL-NOT: {{.*}} = llvm.mlir.constant(1.8{{.*}}e+01 : f32)
  !$omp target map(tofrom:v)
  v = 18.0
  !$omp end target
  go to 20
20 continue
  ! Second unreachable block (only reachable from first unreachable block)
  ! ALL-NOT: {{.*}} = llvm.mlir.constant(1.9{{.*}}e+01 : f32)
  !$omp target map(tofrom:v)
  v = 19.0
  !$omp end target
30 continue
  ! Reachable from entry - target should remain
  !$omp target map(tofrom:v)
  ! ALL: omp.target
  ! ALL: {{.*}} = llvm.mlir.constant(2.0{{.*}}e+01 : f32)
  v = 20.0
  !$omp end target
  ! Expect exactly 1 omp.target in function
  ! ALL-NOT: omp.target
  ! ALL: llvm.return
end subroutine

! Test 14: Both branches reachable - all targets should remain
! ALL-LABEL: llvm.func @_QPtest_both_branches_reachable
! ALL: omp.target
! ALL: omp.target
! ALL: llvm.return
subroutine test_both_branches_reachable(cond)
  logical, intent(in) :: cond
  real :: v
  if (cond) then
    !$omp target map(tofrom:v)
    v = 21.0
    !$omp end target
  else
    !$omp target map(tofrom:v)
    v = 22.0
    !$omp end target
  end if
end subroutine
