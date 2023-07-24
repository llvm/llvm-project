! This test checks lowering of OpenACC reduction clause.

! RUN: bbc -fopenacc -emit-fir %s -o - | FileCheck %s

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_ref_z32 : !fir.ref<!fir.complex<4>> reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.complex<4>>):
! CHECK:   %[[REAL:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:   %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[UNDEF:.*]] = fir.undefined !fir.complex<4>
! CHECK:   %[[UNDEF1:.*]] = fir.insert_value %[[UNDEF]], %[[REAL]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:   %[[UNDEF2:.*]] = fir.insert_value %[[UNDEF1]], %[[IMAG]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.complex<4>
! CHECK:   fir.store %[[UNDEF2]] to %[[ALLOCA]] : !fir.ref<!fir.complex<4>>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.complex<4>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.complex<4>>, %[[ARG1:.*]]: !fir.ref<!fir.complex<4>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.complex<4>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.complex<4>>
! CHECK:   %[[COMBINED:.*]] = fir.mulc %[[LOAD0]], %[[LOAD1]] : !fir.complex<4>
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<!fir.complex<4>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.complex<4>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_z32 : !fir.ref<!fir.complex<4>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.complex<4>>):
! CHECK:   %[[REAL:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[UNDEF:.*]] = fir.undefined !fir.complex<4>
! CHECK:   %[[UNDEF1:.*]] = fir.insert_value %[[UNDEF]], %[[REAL]], [0 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:   %[[UNDEF2:.*]] = fir.insert_value %[[UNDEF1]], %[[IMAG]], [1 : index] : (!fir.complex<4>, f32) -> !fir.complex<4>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.complex<4>
! CHECK:   fir.store %[[UNDEF2]] to %[[ALLOCA]] : !fir.ref<!fir.complex<4>>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.complex<4>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.complex<4>>, %[[ARG1:.*]]: !fir.ref<!fir.complex<4>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.complex<4>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.complex<4>>
! CHECK:   %[[COMBINED:.*]] = fir.addc %[[LOAD0]], %[[LOAD1]] : !fir.complex<4>
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<!fir.complex<4>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.complex<4>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_neqv_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <neqv> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[CST:.*]] = arith.constant false
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.logical<4>
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.logical<4>>, %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[CONV0:.*]] = fir.convert %[[LOAD0]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CONV1:.*]] = fir.convert %[[LOAD1]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CMP:.*]] = arith.cmpi ne, %[[CONV0]], %[[CONV1]] : i1
! CHECK:   %[[CMP_CONV:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CMP_CONV]] to %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_eqv_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <eqv> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[CST:.*]] = arith.constant true
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.logical<4>
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.logical<4>>, %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[CONV0:.*]] = fir.convert %[[LOAD0]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CONV1:.*]] = fir.convert %[[LOAD1]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CMP:.*]] = arith.cmpi eq, %[[CONV0]], %[[CONV1]] : i1
! CHECK:   %[[CMP_CONV:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CMP_CONV]] to %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_lor_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <lor> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[CST:.*]] = arith.constant false
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.logical<4>
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.logical<4>>, %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[CONV0:.*]] = fir.convert %[[LOAD0]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CONV1:.*]] = fir.convert %[[LOAD1]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CMP:.*]] = arith.ori %[[CONV0]], %[[CONV1]] : i1
! CHECK:   %[[CMP_CONV:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CMP_CONV]] to %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_land_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <land> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[CST:.*]] = arith.constant true
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.logical<4>
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.logical<4>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<!fir.logical<4>>, %[[ARG1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<!fir.logical<4>>
! CHECK:   %[[CONV0:.*]] = fir.convert %[[LOAD0]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CONV1:.*]] = fir.convert %[[LOAD1]] : (!fir.logical<4>) -> i1
! CHECK:   %[[CMP:.*]] = arith.andi %[[CONV0]], %[[CONV1]] : i1
! CHECK:   %[[CMP_CONV:.*]] = fir.convert %[[CMP]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CMP_CONV]] to %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.logical<4>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_xor_ref_i32 : !fir.ref<i32> reduction_operator <xor> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[CST:.*]] = arith.constant 0 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   fir.store %[[CST]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[COMBINED:.*]] = arith.xori %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_ior_ref_i32 : !fir.ref<i32> reduction_operator <ior> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[CST:.*]] = arith.constant 0 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   fir.store %[[CST]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[COMBINED:.*]] = arith.ori %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_iand_ref_i32 : !fir.ref<i32> reduction_operator <iand> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[CST:.*]] = arith.constant -1 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   fir.store %[[CST]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[COMBINED:.*]] = arith.andi %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <max> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[INIT:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   %[[LB:.*]] = arith.constant 0 : index
! CHECK:   %[[UB:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
! CHECK:     %[[COORD:.*]] = fir.coordinate_of %[[ALLOCA]], %[[IV]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:     fir.store %[[INIT]] to %[[COORD]] : !fir.ref<f32>
! CHECK:   }
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xf32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_max_ref_f32 : !fir.ref<f32> reduction_operator <max> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
! CHECK:   %[[INIT:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca f32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[CMP:.*]] = arith.cmpf ogt, %[[LOAD0]], %[[LOAD1]] : f32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD0]], %[[LOAD1]] : f32
! CHECK:   fir.store %[[SELECT]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_ref_100x10xi32 : !fir.ref<!fir.array<100x10xi32>> reduction_operator <max> init {
! CHECK: ^bb0(%arg0: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant -2147483648 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10xi32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100x10xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_max_ref_i32 : !fir.ref<i32> reduction_operator <max> init {
! CHECK: ^bb0(%arg0: !fir.ref<i32>):
! CHECK:   %[[INIT:.*]] = arith.constant -2147483648 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[CMP:.*]] = arith.cmpi sgt, %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[SELECT]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_min_ref_100x10xf32 : !fir.ref<!fir.array<100x10xf32>> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10xf32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 3.40282347E+38 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10xf32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100x10xf32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_min_ref_f32 : !fir.ref<f32> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
! CHECK:   %[[INIT:.*]] = arith.constant 3.40282347E+38 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca f32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[CMP:.*]] = arith.cmpf olt, %[[LOAD0]], %[[LOAD1]] : f32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD0]], %[[LOAD1]] : f32
! CHECK:   fir.store %[[SELECT]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_min_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 2147483647 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_min_ref_i32 : !fir.ref<i32> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[INIT:.*]] = arith.constant 2147483647 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[CMP:.*]] = arith.cmpi slt, %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[SELECT]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_ref_f32 : !fir.ref<f32> reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
! CHECK:   %[[INIT:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca f32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[COMBINED:.*]] = arith.mulf %[[LOAD0]], %[[LOAD1]] fastmath<contract> : f32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 1 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_ref_i32 : !fir.ref<i32> reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[INIT:.*]] = arith.constant 1 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[COMBINED:.*]] = arith.muli %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xf32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_f32 : !fir.ref<f32> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
! CHECK:   %[[INIT:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca f32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[COMBINED:.*]] = arith.addf %[[LOAD0]], %[[LOAD1]] fastmath<contract> : f32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_100x10x2xi32 : !fir.ref<!fir.array<100x10x2xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10x2xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10x2xi32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100x10x2xi32>>
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
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10xi32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100x10xi32>>
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
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<!fir.array<100xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_i32 : !fir.ref<i32> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<i32>):
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca i32
! CHECK:   fir.store %[[INIT]] to %[[ALLOCA]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ALLOCA]] : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[COMBINED:.*]] = arith.addi %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
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
! CHECK:       acc.loop reduction(@reduction_add_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       acc.loop reduction(@reduction_add_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK:       acc.loop reduction(@reduction_mul_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       acc.loop reduction(@reduction_mul_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK:       acc.loop reduction(@reduction_min_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       acc.loop reduction(@reduction_min_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK:       acc.loop reduction(@reduction_max_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       acc.loop reduction(@reduction_max_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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

subroutine acc_reduction_iand()
  integer :: i
  !$acc parallel reduction(iand:i)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_iand()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<i32>)   -> !fir.ref<i32> {name = "i"}
! CHECK: acc.parallel   reduction(@reduction_iand_ref_i32 -> %[[RED]] : !fir.ref<i32>)

subroutine acc_reduction_ior()
  integer :: i
  !$acc parallel reduction(ior:i)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_ior()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<i32>)   -> !fir.ref<i32> {name = "i"}
! CHECK: acc.parallel reduction(@reduction_ior_ref_i32 -> %[[RED]] : !fir.ref<i32>)

subroutine acc_reduction_ieor()
  integer :: i
  !$acc parallel reduction(ieor:i)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_ieor()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<i32>) -> !fir.ref<i32> {name = "i"}
! CHECK: acc.parallel reduction(@reduction_xor_ref_i32 -> %[[RED]] : !fir.ref<i32>)

subroutine acc_reduction_and()
  logical :: l
  !$acc parallel reduction(.and.:l)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_and()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%0 : !fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>> {name = "l"}
! CHECK: acc.parallel reduction(@reduction_land_ref_l32 -> %[[RED]] : !fir.ref<!fir.logical<4>>)

subroutine acc_reduction_or()
  logical :: l
  !$acc parallel reduction(.or.:l)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_or()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>> {name = "l"}
! CHECK: acc.parallel reduction(@reduction_lor_ref_l32 -> %[[RED]] : !fir.ref<!fir.logical<4>>)

subroutine acc_reduction_eqv()
  logical :: l
  !$acc parallel reduction(.eqv.:l)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_eqv()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>> {name = "l"}
! CHECK: acc.parallel reduction(@reduction_eqv_ref_l32 -> %[[RED]] : !fir.ref<!fir.logical<4>>)

subroutine acc_reduction_neqv()
  logical :: l
  !$acc parallel reduction(.neqv.:l)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_neqv()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>> {name = "l"}
! CHECK: acc.parallel reduction(@reduction_neqv_ref_l32 -> %[[RED]] : !fir.ref<!fir.logical<4>>)

subroutine acc_reduction_add_cmplx()
  complex :: c
  !$acc parallel reduction(+:c)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_cmplx()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>> {name = "c"}
! CHECK: acc.parallel reduction(@reduction_add_ref_z32 -> %[[RED]] : !fir.ref<!fir.complex<4>>)

subroutine acc_reduction_mul_cmplx()
  complex :: c
  !$acc parallel reduction(*:c)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_mul_cmplx()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<!fir.complex<4>>) -> !fir.ref<!fir.complex<4>> {name = "c"}
! CHECK: acc.parallel reduction(@reduction_mul_ref_z32 -> %[[RED]] : !fir.ref<!fir.complex<4>>)

subroutine acc_reduction_add_alloc()
  integer, allocatable :: i
  allocate(i)
  !$acc parallel reduction(+:i)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_alloc()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "i", uniq_name = "_QFacc_reduction_add_allocEi"}
! CHECK: %[[LOAD:.*]] = fir.load %[[ALLOCA]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[BOX_ADDR]] : !fir.heap<i32>) -> !fir.heap<i32> {name = "i"}
! CHECK: acc.parallel reduction(@reduction_add_heap_i32 -> %[[RED]] : !fir.heap<i32>)

subroutine acc_reduction_add_pointer(i)
  integer, pointer :: i
  !$acc parallel reduction(+:i)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_pointer(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "i"})
! CHECK: %[[LOAD:.*]] = fir.load %[[ARG0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[LOAD]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[BOX_ADDR]] : !fir.ptr<i32>) -> !fir.ptr<i32> {name = "i"}
! CHECK: acc.parallel reduction(@reduction_add_ptr_i32 -> %[[RED]] : !fir.ptr<i32>)

subroutine acc_reduction_add_static_slice(a)
  integer :: a(100)
  !$acc parallel reduction(+:a(11:20))
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_static_slice(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<100xi32>> {fir.bindc_name = "a"})
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 10 : index
! CHECK: %[[UB:.*]] = arith.constant 19 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[ARG0]] : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a(11:20)"}
! CHECK: acc.parallel reduction(@reduction_add_ref_100xi32 -> %[[RED]] : !fir.ref<!fir.array<100xi32>>)
