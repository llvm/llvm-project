! This test checks lowering of OpenACC reduction clause.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: acc.reduction.recipe @reduction_max_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <max> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<-1.401300e-45> : vector<100xf32>
! CHECK:   acc.yield %[[CST]] : vector<100xf32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP0:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
! CHECK:     %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV0]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV0]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<f32>
! CHECK:     %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<f32>
! CHECK:     %[[CMP:.*]] = arith.cmpf ogt, %[[LOAD1]], %[[LOAD2]] : f32
! CHECK:     %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD1]], %[[LOAD2]] : f32
! CHECK:     fir.store %[[SELECT]] to %[[COORD1]] : !fir.ref<f32>
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100xf32>>
! CHECK: }

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

! CHECK-LABEL: acc.reduction.recipe @reduction_max_ref_100x10xi32 : !fir.ref<!fir.array<100x10xi32>> reduction_operator <max> init {
! CHECK: ^bb0(%arg0: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<-2147483648> : vector<100x10xi32>
! CHECK:   acc.yield %[[CST]] : vector<100x10xi32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100x10xi32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 9 : index
! CHECK:   %[[STEP0:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
! CHECK:     %[[LB1:.*]] = arith.constant 0 : index
! CHECK:     %[[UB1:.*]] = arith.constant 99 : index
! CHECK:     %[[STEP1:.*]] = arith.constant 1 : index
! CHECK:     fir.do_loop %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
! CHECK:       %[[COORD1:.*]] = fir.coordinate_of %[[ARG0:.*]], %[[IV0]], %[[IV1]] : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:       %[[COORD2:.*]] = fir.coordinate_of %[[ARG1:.*]], %[[IV0]], %[[IV1]] : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:       %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<i32>
! CHECK:       %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<i32>
! CHECK:       %[[CMP:.*]] = arith.cmpi sgt, %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:       %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:       fir.store %[[SELECT]] to %[[COORD1]] : !fir.ref<i32>
! CHECK:     }
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100x10xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_min_ref_100x10xf32 : !fir.ref<!fir.array<100x10xf32>> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10xf32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<3.40282347E+38> : vector<100x10xf32>
! CHECK:   acc.yield %[[CST]] : vector<100x10xf32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100x10xf32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10xf32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 9 : index
! CHECK:   %[[STEP0:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
! CHECK:     %[[LB1:.*]] = arith.constant 0 : index
! CHECK:     %[[UB1:.*]] = arith.constant 99 : index
! CHECK:     %[[STEP1:.*]] = arith.constant 1 : index
! CHECK:     fir.do_loop %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
! CHECK:       %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV0]], %[[IV1]] : (!fir.ref<!fir.array<100x10xf32>>, index, index) -> !fir.ref<f32>
! CHECK:       %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV0]], %[[IV1]] : (!fir.ref<!fir.array<100x10xf32>>, index, index) -> !fir.ref<f32>
! CHECK:       %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<f32>
! CHECK:       %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<f32>
! CHECK:       %[[CMP:.*]] = arith.cmpf olt, %[[LOAD1]], %[[LOAD2]] : f32
! CHECK:       %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD1]], %[[LOAD2]] : f32
! CHECK:       fir.store %[[SELECT]] to %[[COORD1]] : !fir.ref<f32>
! CHECK:     }
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100x10xf32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_min_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<2147483647> : vector<100xi32>
! CHECK:   acc.yield %[[CST]] : vector<100xi32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP0:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
! CHECK:     %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV0]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV0]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<i32>
! CHECK:     %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<i32>
! CHECK:     %[[CMP:.*]] = arith.cmpi slt, %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:     %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:     fir.store %[[SELECT]] to %[[COORD1]] : !fir.ref<i32>
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_100x10x2xi32 : !fir.ref<!fir.array<100x10x2xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10x2xi32>>):
! CHECK:   %[[CST:.*]] = arith.constant dense<0> : vector<100x10x2xi32>
! CHECK:   acc.yield %[[CST]] : vector<100x10x2xi32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100x10x2xi32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10x2xi32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 1 : index
! CHECK:   %[[STEP0:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
! CHECK:     %[[LB1:.*]] = arith.constant 0 : index
! CHECK:     %[[UB1:.*]] = arith.constant 9 : index
! CHECK:     %[[STEP1:.*]] = arith.constant 1 : index
! CHECK:     fir.do_loop %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
! CHECK:       %[[LB2:.*]] = arith.constant 0 : index
! CHECK:       %[[UB2:.*]] = arith.constant 99 : index
! CHECK:       %[[STEP2:.*]] = arith.constant 1 : index
! CHECK:       fir.do_loop %[[IV2:.*]] = %[[LB2]] to %[[UB2]] step %[[STEP2]] {
! CHECK:         %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV0]], %[[IV1]], %[[IV2]] : (!fir.ref<!fir.array<100x10x2xi32>>, index, index, index) -> !fir.ref<i32>
! CHECK:         %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV0]], %[[IV1]], %[[IV2]] : (!fir.ref<!fir.array<100x10x2xi32>>, index, index, index) -> !fir.ref<i32>
! CHECK:         %[[LOAD1:.*]] = fir.load %[[COORD1]] : !fir.ref<i32>
! CHECK:         %[[LOAD2:.*]] = fir.load %[[COORD2]] : !fir.ref<i32>
! CHECK:         %[[COMBINED:.*]] = arith.addi %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:         fir.store %[[COMBINED]] to %[[COORD1]] : !fir.ref<i32>
! CHECK:       }
! CHECK:     }
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100x10x2xi32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_100x10xi32 : !fir.ref<!fir.array<100x10xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10xi32>>):
! CHECK: %[[CST:.*]] = arith.constant dense<0> : vector<100x10xi32>
! CHECK: acc.yield %[[CST]] : vector<100x10xi32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.array<100x10xi32>>, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:   %[[LB0:.*]] = arith.constant 0 : index
! CHECK:   %[[UB0:.*]] = arith.constant 9 : index
! CHECK:   %[[STEP0:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV0:.*]] = %[[LB0]] to %[[UB0]] step %[[STEP0]] {
! CHECK:     %[[LB1:.*]] = arith.constant 0 : index
! CHECK:     %[[UB1:.*]] = arith.constant 99 : index
! CHECK:     %[[STEP1:.*]] = arith.constant 1 : index
! CHECK:     fir.do_loop %[[IV1:.*]] = %[[LB1]] to %[[UB1]] step %[[STEP1]] {
! CHECK:       %[[COORD1:.*]] = fir.coordinate_of %[[ARG0]], %[[IV0]], %[[IV1]] : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:       %[[COORD2:.*]] = fir.coordinate_of %[[ARG1]], %[[IV0]], %[[IV1]] : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:       %[[LOAD1]] = fir.load %[[COORD1]] : !fir.ref<i32>
! CHECK:       %[[LOAD2]] = fir.load %[[COORD2]] : !fir.ref<i32>
! CHECK:       %[[COMBINED:.*]] = arith.addi %[[LOAD1]], %[[LOAD2]] : i32
! CHECK:       fir.store %[[COMBINED]] to %[[COORD1]] : !fir.ref<i32>
! CHECK:     }
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100x10xi32>>
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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_add_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_add_ref_100xi32 -> %[[RED_B]] : !fir.ref<!fir.array<100xi32>>)

subroutine acc_reduction_add_int_array_2d(a, b)
  integer :: a(100, 10), b(100, 10)
  integer :: i, j

  !$acc loop collapse(2) reduction(+:b)
  do i = 1, 100
    do j = 1, 10
      b(i, j) = b(i, j) + a(i, j)
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_int_array_2d(
! CHECK-SAME:  %[[ARG0:.*]]: !fir.ref<!fir.array<100x10xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10xi32>> {fir.bindc_name = "b"}) {
! CHECK:       %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[ARG1]] : !fir.ref<!fir.array<100x10xi32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<100x10xi32>> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_add_ref_100x10xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xi32>>) {
! CHECK: } attributes {collapse = 2 : i64}

subroutine acc_reduction_add_int_array_3d(a, b)
  integer :: a(100, 10, 2), b(100, 10, 2)
  integer :: i, j, k

  !$acc loop collapse(3) reduction(+:b)
  do i = 1, 100
    do j = 1, 10
      do k = 1, 2
        b(i, j, k) = b(i, j, k) + a(i, j, k)
      end do
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_int_array_3d(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<100x10x2xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10x2xi32>> {fir.bindc_name = "b"})
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[ARG1]] : !fir.ref<!fir.array<100x10x2xi32>>) bounds(%{{.*}}, %{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<100x10x2xi32>> {name = "b"}
! CHECK: acc.loop reduction(@reduction_add_ref_100x10x2xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10x2xi32>>)
! CHECK: } attributes {collapse = 3 : i64}

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
! CHECK:       acc.loop reduction(@reduction_add_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK: %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_add_ref_100xf32 -> %[[RED_B]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"}
! CHECK:       acc.loop reduction(@reduction_mul_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_mul_ref_100xi32 -> %[[RED_B]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_mul_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<!fir.array<100xf32>>) bounds(%2) -> !fir.ref<!fir.array<100xf32>> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_mul_ref_100xf32 -> %[[RED_B]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_min_i32 -> %[[RED_B]] : !fir.ref<i32>)

subroutine acc_reduction_min_int_array_1d(a, b)
  integer :: a(100), b(100)
  integer :: i

  !$acc loop reduction(min:b)
  do i = 1, 100
    b(i) = min(b(i), a(i))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_min_int_array_1d(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "b"})
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[ARG1]] : !fir.ref<!fir.array<100xi32>>) bounds(%2) -> !fir.ref<!fir.array<100xi32>> {name = "b"} 
! CHECK: acc.loop reduction(@reduction_min_ref_100xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_min_f32 -> %[[RED_B]] : !fir.ref<f32>)

subroutine acc_reduction_min_float_array2d(a, b)
  real :: a(100, 10), b(100, 10)
  integer :: i, j

  !$acc loop reduction(min:b) collapse(2)
  do i = 1, 100
    do j = 1, 10
      b(i, j) = min(b(i, j), a(i, j))
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_min_float_array2d(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<100x10xf32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10xf32>> {fir.bindc_name = "b"})
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[ARG1]] : !fir.ref<!fir.array<100x10xf32>>) bounds(%3, %5) -> !fir.ref<!fir.array<100x10xf32>> {name = "b"} 
! CHECK: acc.loop reduction(@reduction_min_ref_100x10xf32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xf32>>)
! CHECK: attributes {collapse = 2 : i64}

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
! CHECL:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_max_i32 -> %[[RED_B]] : !fir.ref<i32>)

subroutine acc_reduction_max_int_array2d(a, b)
  integer :: a(100, 10), b(100, 10)
  integer :: i, j

  !$acc loop reduction(max:b) collapse(2)
  do i = 1, 100
    do j = 1, 10
      b(i, j) = max(b(i, j), a(i, j))
    end do
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_max_int_array2d(
! CHECK-SAME: %{{.*}}: !fir.ref<!fir.array<100x10xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.array<100x10xi32>> {fir.bindc_name = "b"})
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[ARG1]] : !fir.ref<!fir.array<100x10xi32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<100x10xi32>> {name = "b"} 
! CHECK: acc.loop reduction(@reduction_max_ref_100x10xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xi32>>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[B]] : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_max_f32 -> %[[RED_B]] : !fir.ref<f32>)

subroutine acc_reduction_max_float_array1d(a, b)
  real :: a(100), b(100)
  integer :: i

  !$acc loop reduction(max:b)
  do i = 1, 100
    b(i) = max(b(i), a(i))
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_max_float_array1d(
! CHECK-SAME:  %{{.*}}: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<!fir.array<100xf32>> {fir.bindc_name = "b"})
! CHECK:       %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[ARG1]] : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "b"} 
! CHECK:       acc.loop reduction(@reduction_max_ref_100xf32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100xf32>>) {

