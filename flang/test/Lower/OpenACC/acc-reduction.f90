! This test checks lowering of OpenACC reduction clause.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: acc.reduction.recipe @reduction_max_f32 : f32 reduction_operator <max> init {
! CHECK: ^bb0(%{{.*}}: f32):
! CHECK:   %[[INIT:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   acc.yield %[[INIT]] : f32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
! CHECK:   %[[CMP:.*]] = arith.cmpf ogt, %[[ARG0]], %[[ARG1]] : f32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
! CHECK:   acc.yield %[[SELECT]] : f32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_i32 : i32 reduction_operator <max> init {
! CHECK: ^bb0(%arg0: i32):
! CHECK:   %[[INIT:.*]] = arith.constant -2147483648 : i32
! CHECK:   acc.yield %[[INIT]] : i32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
! CHECK:   %[[CMP:.*]] = arith.cmpi sgt, %[[ARG0]], %[[ARG1]] : i32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : i32
! CHECK:   acc.yield %[[SELECT]] : i32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_min_f32 : f32 reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: f32):
! CHECK:   %[[INIT:.*]] = arith.constant 3.40282347E+38 : f32
! CHECK:   acc.yield %[[INIT]] : f32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
! CHECK:   %[[CMP:.*]] = arith.cmpf olt, %[[ARG0]], %[[ARG1]] : f32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : f32
! CHECK:   acc.yield %[[SELECT]] : f32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_min_i32 : i32 reduction_operator <min> init {
! CHECK: ^bb0(%arg0: i32):
! CHECK:   %[[INIT:.*]] = arith.constant 2147483647 : i32
! CHECK:   acc.yield %[[INIT]] : i32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
! CHECK:   %[[CMP:.*]] = arith.cmpi slt, %[[ARG0]], %[[ARG1]] : i32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[ARG0]], %[[ARG1]] : i32
! CHECK:   acc.yield %[[SELECT]] : i32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_f32 : f32 reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: f32):
! CHECK:   %[[INIT:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:   acc.yield %[[INIT]] : f32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
! CHECK:   %[[COMBINED:.*]] = arith.mulf %[[ARG0]], %[[ARG1]] {{.*}} : f32
! CHECK:   acc.yield %[[COMBINED]] : f32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<1> : vector<100xi32>
! CHECK:   acc.yield %[[CST]] : vector<100xi32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[LB:.*]] = arith.constant 0 : index
! CHECK:   %[[UB:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
! CHECK:     %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<i32>
! CHECK:     %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<i32>
! CHECK:     %[[COMBINED:.*]] = arith.muli %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:     fir.store %[[COMBINED]] to %[[COORD1]] : !fir.ref<i32>
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100xi32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_i32 : i32 reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: i32):
! CHECK:   %[[INIT:.*]] = arith.constant 1 : i32
! CHECK:   acc.yield %[[INIT]] : i32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
! CHECK:   %[[COMBINED:.*]] = arith.muli %[[ARG0]], %[[ARG1]] : i32
! CHECK:   acc.yield %[[COMBINED]] : i32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<0.000000e+00> : vector<100xf32>
! CHECK:   acc.yield %[[CST]] : vector<100xf32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[LB:.*]] = arith.constant 0 : index
! CHECK:   %[[UB:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
! CHECK:   %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:   %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<f32>
! CHECK:   %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<f32>
! CHECK:   %[[COMBINED:.*]] = arith.addf %[[LOAD1]], %[[LOAD2]] fastmath<contract> : f32
! CHECK:   fir.store %[[COMBINED]] to %[[COORD1]] : !fir.ref<f32>
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100xf32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_f32 : f32 reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: f32):
! CHECK:   %[[INIT:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   acc.yield %[[INIT]] : f32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
! CHECK:   %[[COMBINED:.*]] = arith.addf %[[ARG0]], %[[ARG1]] {{.*}} : f32
! CHECK:   acc.yield %[[COMBINED]] : f32
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<0> : vector<100xi32>
! CHECK:   acc.yield %[[CST]] : vector<100xi32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[LB:.*]] = arith.constant 0 : index
! CHECK:   %[[UB:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
! CHECK:     %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<i32>
! CHECK:     %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<i32>
! CHECK:     %[[COMBINED:.*]] = arith.addi %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:     fir.store %[[COMBINED]] to %[[COORD1]] : !fir.ref<i32>
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100xi32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_i32 : i32 reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: i32):
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   acc.yield %[[INIT]] : i32
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
! CHECK:   %[[COMBINED:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
! CHECK:   acc.yield %[[COMBINED]] : i32
! CHECK: }

subroutine acc_reduction_add_int(a, b)
  integer :: a(100)
  integer :: i, b

  !$acc loop reduction(+:b)
  do i = 1, 100
    b = b + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_int(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_add_i32 -> %[[B]] : !fir.ref<i32>)

subroutine acc_reduction_add_int_array_1d(a, b)
  integer :: a(100)
  integer :: i, b(100)

  !$acc loop reduction(+:b)
  do i = 1, 100
    b(i) = b(i) + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_int_array_1d(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_add_ref_100xi32 -> %[[B]] : !fir.ref<!fir.array<100xi32>>)

subroutine acc_reduction_add_float(a, b)
  real :: a(100), b
  integer :: i

  !$acc loop reduction(+:b)
  do i = 1, 100
    b = b + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_float(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_add_f32 -> %[[B]] : !fir.ref<f32>)

subroutine acc_reduction_add_float_array_1d(a, b)
  real :: a(100), b(100)
  integer :: i

  !$acc loop reduction(+:b)
  do i = 1, 100
    b(i) = b(i) + a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_float_array_1d(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_add_ref_100xf32 -> %[[B]] : !fir.ref<!fir.array<100xf32>>)

subroutine acc_reduction_mul_int(a, b)
  integer :: a(100)
  integer :: i, b

  !$acc loop reduction(*:b)
  do i = 1, 100
    b = b * a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_mul_int(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_mul_i32 -> %[[B]] : !fir.ref<i32>)

subroutine acc_reduction_mul_int_array_1d(a, b)
  integer :: a(100)
  integer :: i, b(100)

  !$acc loop reduction(*:b)
  do i = 1, 100
    b(i) = b(i) * a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_mul_int_array_1d(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_mul_ref_100xi32 -> %[[B]] : !fir.ref<!fir.array<100xi32>>)

subroutine acc_reduction_mul_float(a, b)
  real :: a(100), b
  integer :: i

  !$acc loop reduction(*:b)
  do i = 1, 100
    b = b * a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_mul_float(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_mul_f32 -> %[[B]] : !fir.ref<f32>)

subroutine acc_reduction_mul_float_array_1d(a, b)
  real :: a(100), b(100)
  integer :: i

  !$acc loop reduction(*:b)
  do i = 1, 100
    b(i) = b(i) * a(i)
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_mul_float_array_1d(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_mul_ref_100xf32 -> %[[B]] : !fir.ref<!fir.array<100xf32>>)

subroutine acc_reduction_min_int(a, b)
  integer :: a(100)
  integer :: i, b

  !$acc loop reduction(min:b)
  do i = 1, 100
    b = min(b, a(i))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_min_int(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_min_i32 -> %[[B]] : !fir.ref<i32>)

subroutine acc_reduction_min_float(a, b)
  real :: a(100), b
  integer :: i

  !$acc loop reduction(min:b)
  do i = 1, 100
    b = min(b, a(i))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_min_float(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_min_f32 -> %[[B]] : !fir.ref<f32>)

subroutine acc_reduction_max_int(a, b)
  integer :: a(100)
  integer :: i, b

  !$acc loop reduction(max:b)
  do i = 1, 100
    b = max(b, a(i))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_max_int(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<i32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_max_i32 -> %[[B]] : !fir.ref<i32>)

subroutine acc_reduction_max_float(a, b)
  real :: a(100), b
  integer :: i

  !$acc loop reduction(max:b)
  do i = 1, 100
    b = max(b, a(i))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_max_float(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[B:.*]]: !fir.ref<f32> {fir.bindc_name = "b"})
! CHECK:       acc.loop reduction(@reduction_max_f32 -> %[[B]] : !fir.ref<f32>)
