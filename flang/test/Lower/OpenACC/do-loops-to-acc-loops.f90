! This test checks lowering of Fortran do loops and do concurrent loops to OpenACC loop constructs.
! Tests the new functionality that converts Fortran iteration constructs to acc.loop with proper IV handling.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s
! RUN: bbc -fopenacc -emit-hlfir --openacc-do-loop-to-acc-loop=false %s -o - | FileCheck %s --check-prefix=CHECK-NOACCLOOP

! CHECK-LABEL: func.func @_QPbasic_do_loop
subroutine basic_do_loop()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do loop that should be converted to acc.loop
  !$acc kernels
  do i = 1, n
    a(i) = b(i) + 1.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: %[[PRIVATE_IV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_IV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {uniq_name = "_QFbasic_do_loopEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! CHECK-NOACCLOOP-LABEL: func.func @_QPbasic_do_loop
! CHECK-NOACCLOOP: acc.kernels {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_concurrent
subroutine basic_do_concurrent()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do concurrent loop
  !$acc kernels
  do concurrent (i = 1:n)
    a(i) = b(i) + 1.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: %[[PRIVATE_IV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_IV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {uniq_name = "_QFbasic_do_concurrentEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! CHECK-NOACCLOOP-LABEL: func.func @_QPbasic_do_concurrent
! CHECK-NOACCLOOP: acc.kernels {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_loop_parallel
subroutine basic_do_loop_parallel()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do loop with acc parallel that should be converted to acc.loop
  !$acc parallel
  do i = 1, n
    a(i) = b(i) + 1.0
  end do
  !$acc end parallel

! CHECK: acc.parallel {
! CHECK: %[[PRIVATE_IV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_IV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {uniq_name = "_QFbasic_do_loop_parallelEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! CHECK-NOACCLOOP-LABEL: func.func @_QPbasic_do_loop_parallel
! CHECK-NOACCLOOP: acc.parallel {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_loop_serial
subroutine basic_do_loop_serial()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do loop with acc serial that should be converted to acc.loop
  !$acc serial
  do i = 1, n
    a(i) = b(i) + 1.0
  end do
  !$acc end serial

! CHECK: acc.serial {
! CHECK: %[[PRIVATE_IV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_IV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {uniq_name = "_QFbasic_do_loop_serialEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}

! CHECK-NOACCLOOP-LABEL: func.func @_QPbasic_do_loop_serial
! CHECK-NOACCLOOP: acc.serial {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_concurrent_parallel
subroutine basic_do_concurrent_parallel()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do concurrent loop with acc parallel
  !$acc parallel
  do concurrent (i = 1:n)
    a(i) = b(i) + 1.0
  end do
  !$acc end parallel

! CHECK: acc.parallel {
! CHECK: %[[PRIVATE_IV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_IV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {uniq_name = "_QFbasic_do_concurrent_parallelEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

! CHECK-NOACCLOOP-LABEL: func.func @_QPbasic_do_concurrent_parallel
! CHECK-NOACCLOOP: acc.parallel {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPbasic_do_concurrent_serial
subroutine basic_do_concurrent_serial()
  integer :: i
  integer, parameter :: n = 10
  real, dimension(n) :: a, b

  ! Basic do concurrent loop with acc serial
  !$acc serial
  do concurrent (i = 1:n)
    a(i) = b(i) + 1.0
  end do
  !$acc end serial

! CHECK: acc.serial {
! CHECK: %[[PRIVATE_IV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_IV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {uniq_name = "_QFbasic_do_concurrent_serialEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {inclusiveUpperbound = array<i1: true>, seq = [#acc.device_type<none>]}

! CHECK-NOACCLOOP-LABEL: func.func @_QPbasic_do_concurrent_serial
! CHECK-NOACCLOOP: acc.serial {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPmulti_dimension_do_concurrent
subroutine multi_dimension_do_concurrent()
  integer :: i, j, k
  integer, parameter :: n = 10, m = 20, l = 5
  real, dimension(n,m,l) :: a, b

  ! Multi-dimensional do concurrent with multiple iteration variables
  !$acc kernels
  do concurrent (i = 1:n, j = 1:m, k = 1:l)
    a(i,j,k) = b(i,j,k) * 2.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK-DAG: %[[PRIVATE_I:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK-DAG: %[[PRIVATE_J:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK-DAG: %[[PRIVATE_K:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "k"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_I]] : !fir.ref<i32>, @privatization_ref_i32 -> %[[PRIVATE_J]] : !fir.ref<i32>, @privatization_ref_i32 -> %[[PRIVATE_K]] : !fir.ref<i32>) control(%{{.*}} : i32, %{{.*}} : i32, %{{.*}} : i32) = (%c1{{.*}}, %c1{{.*}}, %c1{{.*}} : i32, i32, i32) to (%{{.*}}, %{{.*}}, %{{.*}} : i32, i32, i32) step (%c1{{.*}}, %c1{{.*}}, %c1{{.*}} : i32, i32, i32)
! CHECK-DAG: %[[PRIVATE_I_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_I]] {uniq_name = "_QFmulti_dimension_do_concurrentEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG: %[[PRIVATE_J_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_J]] {uniq_name = "_QFmulti_dimension_do_concurrentEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK-DAG: %[[PRIVATE_K_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_K]] {uniq_name = "_QFmulti_dimension_do_concurrentEk"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_I_DECLARE]]#0 : !fir.ref<i32>
! CHECK: fir.store %{{.*}} to %[[PRIVATE_J_DECLARE]]#0 : !fir.ref<i32>
! CHECK: fir.store %{{.*}} to %[[PRIVATE_K_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_J_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_K_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_J_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_K_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true, true, true>}

! CHECK-NOACCLOOP-LABEL: func.func @_QPmulti_dimension_do_concurrent
! CHECK-NOACCLOOP: acc.kernels {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine


! CHECK-LABEL: func.func @_QPnested_do_loops
subroutine nested_do_loops()
  integer :: i, j
  integer, parameter :: n = 10, m = 20
  real, dimension(n,m) :: a, b

  ! Nested do loops
  !$acc kernels
  do i = 1, n
    do j = 1, m
      a(i,j) = b(i,j) + i + j
    end do
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK-DAG: %[[PRIVATE_I:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_I]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK-DAG: %[[PRIVATE_I_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_I]] {uniq_name = "_QFnested_do_loopsEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_I_DECLARE]]#0 : !fir.ref<i32>
! CHECK-DAG: %[[PRIVATE_J:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_J]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK-DAG: %[[PRIVATE_J_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_J]] {uniq_name = "_QFnested_do_loopsEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_J_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_J_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_J_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! CHECK-NOACCLOOP-LABEL: func.func @_QPnested_do_loops
! CHECK-NOACCLOOP: acc.kernels {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPvariable_bounds_and_step
subroutine variable_bounds_and_step(n, start_val, step_val)
  integer, intent(in) :: n, start_val, step_val
  integer :: i
  real, dimension(n) :: a, b

  ! Do loop with variable bounds and step
  !$acc kernels
  do i = start_val, n, step_val
    a(i) = b(i) * 2.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: %[[PRIVATE_IV:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_IV]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_IV]] {uniq_name = "_QFvariable_bounds_and_stepEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}

! CHECK-NOACCLOOP-LABEL: func.func @_QPvariable_bounds_and_step
! CHECK-NOACCLOOP: acc.kernels {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPdifferent_iv_types
subroutine different_iv_types()
  integer(kind=8) :: i8
  integer(kind=4) :: i4
  integer(kind=2) :: i2
  integer, parameter :: n = 10
  real, dimension(n) :: a, b, c, d

  ! Test different iteration variable types
  !$acc kernels
  do i8 = 1_8, int(n,8)
    a(i8) = b(i8) + 1.0
  end do
  !$acc end kernels

  !$acc kernels
  do i4 = 1, n
    b(i4) = c(i4) + 1.0
  end do
  !$acc end kernels

  !$acc kernels
  do i2 = 1_2, int(n,2)
    c(i2) = d(i2) + 1.0
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK: %[[PRIVATE_I8:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i64>) -> !fir.ref<i64> {implicit = true, name = "i8"}
! CHECK: acc.loop private(@privatization_ref_i64 -> %[[PRIVATE_I8]] : !fir.ref<i64>) control(%{{.*}} : i64) = (%{{.*}} : i64) to (%{{.*}} : i64) step (%{{.*}} : i64)
! CHECK: %[[PRIVATE_I8_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_I8]] {uniq_name = "_QFdifferent_iv_typesEi8"} : (!fir.ref<i64>) -> (!fir.ref<i64>, !fir.ref<i64>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_I8_DECLARE]]#0 : !fir.ref<i64>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I8_DECLARE]]#0 : !fir.ref<i64>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I8_DECLARE]]#0 : !fir.ref<i64>
! CHECK: acc.kernels {
! CHECK: %[[PRIVATE_I4:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i4"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_I4]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_I4_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_I4]] {uniq_name = "_QFdifferent_iv_typesEi4"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_I4_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I4_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I4_DECLARE]]#0 : !fir.ref<i32>
! CHECK: acc.kernels {
! CHECK: %[[PRIVATE_I2:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i16>) -> !fir.ref<i16> {implicit = true, name = "i2"}
! CHECK: acc.loop private(@privatization_ref_i16 -> %[[PRIVATE_I2]] : !fir.ref<i16>) control(%{{.*}} : i16) = (%{{.*}} : i16) to (%{{.*}} : i16) step (%{{.*}} : i16)
! CHECK: %[[PRIVATE_I2_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_I2]] {uniq_name = "_QFdifferent_iv_typesEi2"} : (!fir.ref<i16>) -> (!fir.ref<i16>, !fir.ref<i16>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_I2_DECLARE]]#0 : !fir.ref<i16>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I2_DECLARE]]#0 : !fir.ref<i16>
! CHECK: %{{.*}} = fir.load %[[PRIVATE_I2_DECLARE]]#0 : !fir.ref<i16>

! CHECK-NOACCLOOP-LABEL: func.func @_QPdifferent_iv_types
! CHECK-NOACCLOOP: acc.kernels {
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPnested_loop_with_reduction
subroutine nested_loop_with_reduction(x, y)
  integer :: x, y
  integer :: i, j

  ! Nested loop with reduction variables - check that reduction operations
  ! are correctly scoped (outer loop reduction should not be inside inner loop)
  !$acc parallel
  !$acc loop reduction(+:x,y)
  do i = 1, 10
    do j = 1, 20
      y = y + 1
    end do
    x = x + 1
  end do
  !$acc end parallel

! CHECK: acc.parallel {
! CHECK: %[[REDUCTION_X:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {name = "x"}
! CHECK: %[[REDUCTION_Y:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {name = "y"}
! CHECK: %[[PRIVATE_I:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_I]] : !fir.ref<i32>) reduction(@reduction_add_ref_i32 -> %[[REDUCTION_X]] : !fir.ref<i32>, @reduction_add_ref_i32 -> %[[REDUCTION_Y]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_I_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_I]] {uniq_name = "_QFnested_loop_with_reductionEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_I_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %[[PRIVATE_J:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {implicit = true, name = "j"}
! CHECK: acc.loop private(@privatization_ref_i32 -> %[[PRIVATE_J]] : !fir.ref<i32>) control(%{{.*}} : i32) = (%{{.*}} : i32) to (%{{.*}} : i32) step (%{{.*}} : i32)
! CHECK: %[[PRIVATE_J_DECLARE:.*]]:2 = hlfir.declare %[[PRIVATE_J]] {uniq_name = "_QFnested_loop_with_reductionEj"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: fir.store %{{.*}} to %[[PRIVATE_J_DECLARE]]#0 : !fir.ref<i32>
! CHECK: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
! CHECK: hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {auto_ = [#acc.device_type<none>], inclusiveUpperbound = array<i1: true>}
! CHECK: %{{.*}} = fir.load %{{.*}} : !fir.ref<i32>
! CHECK: %{{.*}} = arith.addi %{{.*}}, %{{.*}} : i32
! CHECK: hlfir.assign %{{.*}} to %{{.*}} : i32, !fir.ref<i32>
! CHECK: acc.yield
! CHECK: attributes {inclusiveUpperbound = array<i1: true>, independent = [#acc.device_type<none>]}

! CHECK-NOACCLOOP-LABEL: func.func @_QPnested_loop_with_reduction
! CHECK-NOACCLOOP: acc.parallel {
! CHECK-NOACCLOOP: acc.loop{{.*}}reduction{{.*}}control
! CHECK-NOACCLOOP-NOT: acc.loop

end subroutine

! -----------------------------------------------------------------------------------------
! Tests for loops that should NOT be converted to acc.loop due to unstructured control flow

! CHECK-LABEL: func.func @_QPinfinite_loop_no_iv
subroutine infinite_loop_no_iv()
  integer :: i
  logical :: condition

  ! Infinite loop with no induction variable - should NOT convert to acc.loop
  !$acc kernels
  do
    i = i + 1
    if (i > 100) exit
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK-NOT: acc.loop

end subroutine

! CHECK-LABEL: func.func @_QPwhile_like_loop
subroutine while_like_loop()
  integer :: i
  logical :: condition

  i = 1
  condition = .true.

  ! While-like infinite loop - should NOT convert to acc.loop
  !$acc kernels
  do while (condition)
    i = i + 1
    if (i > 100) condition = .false.
  end do
  !$acc end kernels

! CHECK: acc.kernels {
! CHECK-NOT: acc.loop

end subroutine
