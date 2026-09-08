! RUN: %flang_fc1 -emit-hlfir -fopenmp -fopenmp-version=51 -o - %s 2>&1 | FileCheck %s

subroutine constant_ascending()
  integer :: i
  !$omp unroll
  do i = 1, 100
  end do
  !$omp end unroll
end subroutine

! CHECK-LABEL: func.func @_QPconstant_ascending()
! CHECK-NOT: arith.cmpi
! CHECK-NOT: arith.select
! CHECK-NOT: arith.subi
! CHECK-NOT: arith.divui
! CHECK-NOT: arith.addi
! CHECK: %[[TRIP_COUNT:.*]] = arith.constant 100 : i32
! CHECK-NEXT: %[[CLI:.*]] = omp.new_cli
! CHECK-NEXT: omp.canonical_loop(%[[CLI]]) {{.*}} : i32 in range(%[[TRIP_COUNT]]) {

subroutine constant_descending()
  integer :: i
  !$omp unroll
  do i = 100, 1, -1
  end do
  !$omp end unroll
end subroutine

! CHECK-LABEL: func.func @_QPconstant_descending()
! CHECK-NOT: arith.cmpi
! CHECK-NOT: arith.select
! CHECK-NOT: arith.subi
! CHECK-NOT: arith.divui
! CHECK-NOT: arith.addi
! CHECK: arith.constant -1 : i32
! CHECK: %[[TRIP_COUNT:.*]] = arith.constant 100 : i32
! CHECK-NEXT: %[[CLI:.*]] = omp.new_cli
! CHECK-NEXT: omp.canonical_loop(%[[CLI]]) {{.*}} : i32 in range(%[[TRIP_COUNT]]) {

subroutine constant_zero_trip()
  integer :: i
  !$omp unroll
  do i = 100, 1
  end do
  !$omp end unroll
end subroutine

! CHECK-LABEL: func.func @_QPconstant_zero_trip()
! CHECK-NOT: arith.cmpi
! CHECK-NOT: arith.select
! CHECK-NOT: arith.subi
! CHECK-NOT: arith.divui
! CHECK-NOT: arith.addi
! CHECK: %[[TRIP_COUNT:.*]] = arith.constant 0 : i32
! CHECK-NEXT: %[[CLI:.*]] = omp.new_cli
! CHECK-NEXT: omp.canonical_loop(%[[CLI]]) {{.*}} : i32 in range(%[[TRIP_COUNT]]) {

subroutine constant_non_unit_step()
  integer :: i
  !$omp unroll
  do i = 1, 100, 3
  end do
  !$omp end unroll
end subroutine

! CHECK-LABEL: func.func @_QPconstant_non_unit_step()
! CHECK-NOT: arith.cmpi
! CHECK-NOT: arith.select
! CHECK-NOT: arith.subi
! CHECK-NOT: arith.divui
! CHECK-NOT: arith.addi
! CHECK: %[[TRIP_COUNT:.*]] = arith.constant 34 : i32
! CHECK-NEXT: %[[CLI:.*]] = omp.new_cli
! CHECK-NEXT: omp.canonical_loop(%[[CLI]]) {{.*}} : i32 in range(%[[TRIP_COUNT]]) {

subroutine runtime_bounds(lb, ub, step)
  integer :: i, lb, ub, step
  !$omp unroll
  do i = lb, ub, step
  end do
  !$omp end unroll
end subroutine

! CHECK-LABEL: func.func @_QPruntime_bounds
! CHECK: %[[LB:.*]] = fir.load {{.*}} : !fir.ref<i32>
! CHECK-NEXT: %[[UB:.*]] = fir.load {{.*}} : !fir.ref<i32>
! CHECK-NEXT: %[[STEP:.*]] = fir.load {{.*}} : !fir.ref<i32>
! CHECK: %[[ZERO:.*]] = arith.constant 0 : i32
! CHECK-NEXT: %[[ONE:.*]] = arith.constant 1 : i32
! CHECK-NEXT: %[[IS_DOWNWARDS:.*]] = arith.cmpi slt, %[[STEP]], %[[ZERO]] : i32
! CHECK-NEXT: %[[NEG_STEP:.*]] = arith.subi %[[ZERO]], %[[STEP]] : i32
! CHECK-NEXT: %[[INCR:.*]] = arith.select %[[IS_DOWNWARDS]], %[[NEG_STEP]], %[[STEP]] : i32
! CHECK-NEXT: %[[LOWER:.*]] = arith.select %[[IS_DOWNWARDS]], %[[UB]], %[[LB]] : i32
! CHECK-NEXT: %[[UPPER:.*]] = arith.select %[[IS_DOWNWARDS]], %[[LB]], %[[UB]] : i32
! CHECK-NEXT: %[[SPAN:.*]] = arith.subi %[[UPPER]], %[[LOWER]] overflow<nuw> : i32
! CHECK-NEXT: %[[TC_MINUS_ONE:.*]] = arith.divui %[[SPAN]], %[[INCR]] : i32
! CHECK-NEXT: %[[TC_IF_LOOPING:.*]] = arith.addi %[[TC_MINUS_ONE]], %[[ONE]] overflow<nuw> : i32
! CHECK-NEXT: %[[IS_ZERO_TC:.*]] = arith.cmpi slt, %[[UPPER]], %[[LOWER]] : i32
! CHECK-NEXT: %[[TRIP_COUNT:.*]] = arith.select %[[IS_ZERO_TC]], %[[ZERO]], %[[TC_IF_LOOPING]] : i32
! CHECK-NEXT: %[[CLI:.*]] = omp.new_cli
! CHECK-NEXT: omp.canonical_loop(%[[CLI]]) {{.*}} : i32 in range(%[[TRIP_COUNT]]) {

subroutine constant_i64()
  integer(kind=8) :: i
  !$omp unroll
  do i = 1_8, 100_8
  end do
  !$omp end unroll
end subroutine

! CHECK-LABEL: func.func @_QPconstant_i64()
! CHECK-NOT: arith.cmpi
! CHECK-NOT: arith.select
! CHECK-NOT: arith.subi
! CHECK-NOT: arith.divui
! CHECK-NOT: arith.addi
! CHECK: %[[TRIP_COUNT:.*]] = arith.constant 100 : i64
! CHECK-NEXT: %[[CLI:.*]] = omp.new_cli
! CHECK-NEXT: omp.canonical_loop(%[[CLI]]) {{.*}} : i64 in range(%[[TRIP_COUNT]]) {

subroutine constant_i64_default_kind_bounds()
  integer(kind=8) :: i
  !$omp unroll
  do i = 1, 100
  end do
  !$omp end unroll
end subroutine

! CHECK-LABEL: func.func @_QPconstant_i64_default_kind_bounds()
! CHECK-NOT: fir.convert
! CHECK-NOT: arith.cmpi
! CHECK-NOT: arith.select
! CHECK-NOT: arith.subi
! CHECK-NOT: arith.divui
! CHECK-NOT: arith.addi
! CHECK: %[[TRIP_COUNT:.*]] = arith.constant 100 : i64
! CHECK-NEXT: %[[CLI:.*]] = omp.new_cli
! CHECK-NEXT: omp.canonical_loop(%[[CLI]]) {{.*}} : i64 in range(%[[TRIP_COUNT]]) {
