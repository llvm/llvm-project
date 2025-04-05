! This test checks lowering of OpenACC reduction clause.

! RUN: bbc -fopenacc -emit-hlfir --openacc-unwrap-fir-box=true --openacc-generate-default-bounds=true %s -o - | FileCheck %s

! CHECK-LABEL: acc.reduction.recipe @reduction_max_box_UxUxf32 : !fir.box<!fir.array<?x?xf32>> reduction_operator <max> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?x?xf32>>):
! CHECK:   %[[CST:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   %[[DIMS0:.*]]:3 = fir.box_dims %[[ARG0]], %c0{{.*}} : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:   %[[DIMS1:.*]]:3 = fir.box_dims %[[ARG0]], %c1 : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[DIMS0]]#1, %[[DIMS1]]#1 : (index, index) -> !fir.shape<2>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[DIMS0]]#1, %[[DIMS1]]#1 {bindc_name = ".tmp", uniq_name = ""} 
! CHECK:   %[[DECL:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.heap<!fir.array<?x?xf32>>)
! CHECK:   hlfir.assign %[[CST]] to %[[DECL]]#0 : f32, !fir.box<!fir.array<?x?xf32>>
! CHECK:   acc.yield %[[DECL]]#0 : !fir.box<!fir.array<?x?xf32>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[V1:.*]]: !fir.box<!fir.array<?x?xf32>>, %[[V2:.*]]: !fir.box<!fir.array<?x?xf32>>, %[[LB0:.*]]: index, %[[UB0:.*]]: index, %[[STEP0:.*]]: index, %[[LB1:.*]]: index, %[[UB1:.*]]: index, %[[STEP1:.*]]: index):
 
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[DES_V1:.*]] = hlfir.designate %[[V1]] (%[[LB0]]:%[[UB0]]:%[[STEP0]], %[[LB1]]:%[[UB1]]:%[[STEP1]])  shape %[[SHAPE]] : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:   %[[DES_V2:.*]] = hlfir.designate %[[V2]] (%[[LB0]]:%[[UB0]]:%[[STEP0]], %[[LB1]]:%[[UB1]]:%[[STEP1]])  shape %[[SHAPE]] : (!fir.box<!fir.array<?x?xf32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.box<!fir.array<?x?xf32>>
! CHECK:   %[[ELEMENTAL:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<2>) -> !hlfir.expr<?x?xf32> {
! CHECK:   ^bb0(%[[ARG0:.*]]: index, %[[ARG1:.*]]: index):
! CHECK:     %[[D1:.*]] = hlfir.designate %[[DES_V1]] (%[[ARG0]], %[[ARG1]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
! CHECK:     %[[D2:.*]] = hlfir.designate %[[DES_V2]] (%[[ARG0]], %[[ARG1]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
! CHECK:     %[[LOAD1:.*]] = fir.load %[[D1]] : !fir.ref<f32>
! CHECK:     %[[LOAD2:.*]] = fir.load %[[D2]] : !fir.ref<f32>
! CHECK:     %[[CMP:.*]] = arith.cmpf ogt, %[[LOAD1]], %[[LOAD2]] {{.*}} : f32
! CHECK:     %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD1]], %[[LOAD2]] : f32
! CHECK:     hlfir.yield_element %[[SELECT]] : f32
! CHECK:   }
! CHECK:   hlfir.assign %[[ELEMENTAL]] to %[[V1]] : !hlfir.expr<?x?xf32>, !fir.box<!fir.array<?x?xf32>>
! CHECK:   acc.yield %[[V1]] : !fir.box<!fir.array<?x?xf32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_box_ptr_Uxf32 : !fir.box<!fir.ptr<!fir.array<?xf32>>> reduction_operator <max> init {
! CHECK: ^bb0(%{{.*}}: !fir.box<!fir.ptr<!fir.array<?xf32>>>):
! CHECK: } combiner {
! CHECK: ^bb0(%{{.*}}: !fir.box<!fir.ptr<!fir.array<?xf32>>>, %{{.*}}: !fir.box<!fir.ptr<!fir.array<?xf32>>>, %{{.*}}: index, %{{.*}}: index, %{{.*}}: index):
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_box_heap_Uxf32 : !fir.box<!fir.heap<!fir.array<?xf32>>> reduction_operator <max> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.heap<!fir.array<?xf32>>>):
! CHECK:   %[[CST:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xf32>, %[[BOX_DIMS]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %2(%1) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
! CHECK:   hlfir.assign %[[CST]] to %[[DECLARE]]#0 : f32, !fir.box<!fir.array<?xf32>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xf32>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.heap<!fir.array<?xf32>>>, %[[ARG1:.*]]: !fir.box<!fir.heap<!fir.array<?xf32>>>, %[[ARG2:.*]]: index, %[[ARG3:.*]]: index, %[[ARG4:.*]]: index):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[DES_V1:.*]] = hlfir.designate %[[ARG0]] (%[[ARG2]]:%[[ARG3]]:%[[ARG4]]) shape %[[SHAPE]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:   %[[DES_V2:.*]] = hlfir.designate %[[ARG1]] (%[[ARG2]]:%[[ARG3]]:%[[ARG4]]) shape %[[SHAPE]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:   %[[ELEMENTAL:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:   ^bb0(%[[IV:.*]]: index):
! CHECK:     %[[V1:.*]] = hlfir.designate %[[DES_V1]] (%[[IV]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:     %[[V2:.*]] = hlfir.designate %[[DES_V2]] (%[[IV]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:     %[[LOAD_V1:.*]] = fir.load %[[V1]] : !fir.ref<f32>
! CHECK:     %[[LOAD_V2:.*]] = fir.load %[[V2]] : !fir.ref<f32>
! CHECK:     %[[CMP:.*]] = arith.cmpf ogt, %[[LOAD_V1]], %[[LOAD_V2]] {{.*}} : f32
! CHECK:     %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD_V1]], %[[LOAD_V2]] : f32
! CHECK:     hlfir.yield_element %[[SELECT]] : f32
! CHECK:   }
! CHECK:   hlfir.assign %[[ELEMENTAL]] to %[[ARG0]] : !hlfir.expr<?xf32>, !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:   acc.yield %[[ARG0]] : !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_section_lb1.ub3_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %c0{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:   hlfir.assign %c0{{.*}} to %[[DECLARE]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>, %[[ARG1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[DES1:.*]] = hlfir.designate %[[ARG0]] shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   %[[DES2:.*]] = hlfir.designate %[[ARG1]] shape %[[SHAPE]] : (!fir.box<!fir.array<?xi32>>, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   %[[ELEMENTAL:.*]] = hlfir.elemental %[[SHAPE]] unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:   ^bb0(%[[IV:.*]]: index):
! CHECK:     %[[DES_V1:.*]] = hlfir.designate %[[DES1]] (%[[IV]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[DES_V2:.*]] = hlfir.designate %[[DES2]] (%[[IV]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[LOAD_V1:.*]] = fir.load %[[DES_V1]] : !fir.ref<i32>
! CHECK:     %[[LOAD_V2:.*]] = fir.load %[[DES_V2]] : !fir.ref<i32>
! CHECK:     %[[COMBINED:.*]] = arith.addi %[[LOAD_V1]], %[[LOAD_V2]] : i32
! CHECK:     hlfir.yield_element %[[COMBINED]] : i32
! CHECK:   }
! CHECK:   hlfir.assign %[[ELEMENTAL]] to %[[ARG0]] : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:   acc.yield %[[ARG0]] : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_box_Uxf32 : !fir.box<!fir.array<?xf32>> reduction_operator <max> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xf32>>):
! CHECK:   %[[INIT_VALUE:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xf32>, %0#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
! CHECK:   hlfir.assign %[[INIT_VALUE]] to %[[DECLARE]]#0 : f32, !fir.box<!fir.array<?xf32>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xf32>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xf32>>, %[[ARG1:.*]]: !fir.box<!fir.array<?xf32>>
! CHECK:   %[[LEFT:.*]] = hlfir.designate %[[ARG0]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:   %[[RIGHT:.*]] = hlfir.designate %[[ARG1]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} : (!fir.box<!fir.array<?xf32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xf32>>
! CHECK:   %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xf32> {
! CHECK:   ^bb0(%{{.*}}: index):
! CHECK:     %[[DES_V1:.*]] = hlfir.designate %[[LEFT]] (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[DES_V2:.*]] = hlfir.designate %[[RIGHT]] (%{{.*}})  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:     %[[LOAD_V1:.*]] = fir.load %[[DES_V1]] : !fir.ref<f32>
! CHECK:     %[[LOAD_V2:.*]] = fir.load %[[DES_V2]] : !fir.ref<f32>
! CHECK:     %[[CMPF:.*]] = arith.cmpf ogt, %[[LOAD_V1]], %[[LOAD_V2]] {{.*}} : f32
! CHECK:     %[[SELECT:.*]] = arith.select %[[CMPF]], %[[LOAD_V1]], %[[LOAD_V2]] : f32
! CHECK:     hlfir.yield_element %[[SELECT]] : f32
! CHECK:   }
! CHECK:   hlfir.assign %[[ELEMENTAL]] to %[[ARG0]] : !hlfir.expr<?xf32>, !fir.box<!fir.array<?xf32>>
! CHECK: acc.yield %[[ARG0]] : !fir.box<!fir.array<?xf32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:   %[[INIT_VALUE:.*]] = arith.constant 0 : i32
! CHECK:   %[[C0:.*]] = arith.constant 0 : index
! CHECK:   %[[BOX_DIMS:.*]]:3 = fir.box_dims %[[ARG0]], %[[C0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:   %[[SHAPE:.*]] = fir.shape %[[BOX_DIMS]]#1 : (index) -> !fir.shape<1>
! CHECK:   %[[TEMP:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[TEMP]](%[[SHAPE]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:   hlfir.assign %[[INIT_VALUE]] to %[[DECLARE]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.box<!fir.array<?xi32>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[V1:.*]]: !fir.box<!fir.array<?xi32>>, %[[V2:.*]]: !fir.box<!fir.array<?xi32>>
! CHECK:   %[[LEFT:.*]] = hlfir.designate %[[ARG0]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   %[[RIGHT:.*]] = hlfir.designate %[[ARG1]] (%{{.*}}:%{{.*}}:%{{.*}}) shape %{{.*}} : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:   %[[ELEMENTAL:.*]] = hlfir.elemental %{{.*}} unordered : (!fir.shape<1>) -> !hlfir.expr<?xi32> {
! CHECK:   ^bb0(%{{.*}}: index):
! CHECK:     %[[DES_V1:.*]] = hlfir.designate %[[LEFT]] (%{{.*}})  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[DES_V2:.*]] = hlfir.designate %[[RIGHT]] (%{{.*}})  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:     %[[LOAD_V1:.*]] = fir.load %[[DES_V1]] : !fir.ref<i32>
! CHECK:     %[[LOAD_V2:.*]] = fir.load %[[DES_V2]] : !fir.ref<i32>
! CHECK:     %[[COMBINED:.*]] = arith.addi %[[LOAD_V1]], %[[LOAD_V2]] : i32
! CHECK:     hlfir.yield_element %[[COMBINED]] : i32
! CHECK:   }
! CHECK:   hlfir.assign %[[ELEMENTAL]] to %[[V1]] : !hlfir.expr<?xi32>, !fir.box<!fir.array<?xi32>>
! CHECK:   acc.yield %arg0 : !fir.box<!fir.array<?xi32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_ref_z32 : !fir.ref<complex<f32>> reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<complex<f32>>):
! CHECK:   %[[REAL:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:   %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[UNDEF:.*]] = fir.undefined complex<f32>
! CHECK:   %[[UNDEF1:.*]] = fir.insert_value %[[UNDEF]], %[[REAL]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:   %[[UNDEF2:.*]] = fir.insert_value %[[UNDEF1]], %[[IMAG]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca complex<f32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK:   fir.store %[[UNDEF2]] to %[[DECLARE]]#0 : !fir.ref<complex<f32>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<complex<f32>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<complex<f32>>, %[[ARG1:.*]]: !fir.ref<complex<f32>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<complex<f32>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<complex<f32>>
! CHECK:   %[[COMBINED:.*]] = fir.mulc %[[LOAD0]], %[[LOAD1]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<complex<f32>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<complex<f32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_ref_z32 : !fir.ref<complex<f32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<complex<f32>>):
! CHECK:   %[[REAL:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[IMAG:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[UNDEF:.*]] = fir.undefined complex<f32>
! CHECK:   %[[UNDEF1:.*]] = fir.insert_value %[[UNDEF]], %[[REAL]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:   %[[UNDEF2:.*]] = fir.insert_value %[[UNDEF1]], %[[IMAG]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca complex<f32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK:   fir.store %[[UNDEF2]] to %[[DECLARE]]#0 : !fir.ref<complex<f32>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<complex<f32>>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<complex<f32>>, %[[ARG1:.*]]: !fir.ref<complex<f32>>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<complex<f32>>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<complex<f32>>
! CHECK:   %[[COMBINED:.*]] = fir.addc %[[LOAD0]], %[[LOAD1]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<complex<f32>>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<complex<f32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_neqv_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <neqv> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.logical<4>>):
! CHECK:   %[[CST:.*]] = arith.constant false
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.logical<4>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:   %[[CONVERT:.*]] = fir.convert %[[CST]] : (i1) -> !fir.logical<4>
! CHECK:   fir.store %[[CONVERT]] to %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.logical<4>>
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
! CHECK:   %[[DECLARE]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[CST]] to %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[CST]] to %[[DECLARE:.*]]#0 : !fir.ref<i32>
! CHECK:   acc.yield %[[DECLARE:.*]]#0 : !fir.ref<i32>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[CST]] to %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[COMBINED:.*]] = arith.andi %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_section_ext100_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <max> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[INIT:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:   %[[LB:.*]] = arith.constant 0 : index
! CHECK:   %[[UB:.*]] = arith.constant 99 : index
! CHECK:   %[[STEP:.*]] = arith.constant 1 : index
! CHECK:   fir.do_loop %[[IV:.*]] = %[[LB]] to %[[UB]] step %[[STEP]] {
! CHECK:     %[[COORD:.*]] = fir.coordinate_of %[[DECLARE]]#0, %[[IV]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:     fir.store %[[INIT]] to %[[COORD]] : !fir.ref<f32>
! CHECK:   }
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xf32>>
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
! CHECK:     %[[CMP:.*]] = arith.cmpf ogt, %[[LOAD1]], %[[LOAD2]] {{.*}} : f32
! CHECK:     %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD1]], %[[LOAD2]] : f32
! CHECK:     fir.store %[[SELECT]] to %[[COORD1]] : !fir.ref<f32>
! CHECK:   }
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<!fir.array<100xf32>>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_ref_f32 : !fir.ref<f32> reduction_operator <max> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<f32>):
! CHECK:   %[[INIT:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:   %[[ALLOCA:.*]] = fir.alloca f32
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %0 {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[CMP:.*]] = arith.cmpf ogt, %[[LOAD0]], %[[LOAD1]] {{.*}} : f32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD0]], %[[LOAD1]] : f32
! CHECK:   fir.store %[[SELECT]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_max_section_ext100xext10_ref_100x10xi32 : !fir.ref<!fir.array<100x10xi32>> reduction_operator <max> init {
! CHECK: ^bb0(%arg0: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant -2147483648 : i32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10xi32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x10xi32>>, !fir.ref<!fir.array<100x10xi32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100x10xi32>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[CMP:.*]] = arith.cmpi sgt, %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[SELECT]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_min_section_ext100xext10_ref_100x10xf32 : !fir.ref<!fir.array<100x10xf32>> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10xf32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 3.40282347E+38 : f32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x10xf32>>, !fir.ref<!fir.array<100x10xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100x10xf32>>
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
! CHECK:       %[[CMP:.*]] = arith.cmpf olt, %[[LOAD1]], %[[LOAD2]] {{.*}} : f32
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[CMP:.*]] = arith.cmpf olt, %[[LOAD0]], %[[LOAD1]] {{.*}} : f32
! CHECK:   %[[SELECT:.*]] = arith.select %[[CMP]], %[[LOAD0]], %[[LOAD1]] : f32
! CHECK:   fir.store %[[SELECT]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_min_section_ext100_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <min> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 2147483647 : i32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xi32>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[COMBINED:.*]] = arith.mulf %[[LOAD0]], %[[LOAD1]] fastmath<contract> : f32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_mul_section_ext100_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <mul> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 1 : i32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xi32>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<i32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<i32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<i32>
! CHECK:   %[[COMBINED:.*]] = arith.muli %[[LOAD0]], %[[LOAD1]] : i32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<i32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<i32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_section_ext100_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xf32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xf32>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<f32>
! CHECK: } combiner {
! CHECK: ^bb0(%[[ARG0:.*]]: !fir.ref<f32>, %[[ARG1:.*]]: !fir.ref<f32>):
! CHECK:   %[[LOAD0:.*]] = fir.load %[[ARG0]] : !fir.ref<f32>
! CHECK:   %[[LOAD1:.*]] = fir.load %[[ARG1]] : !fir.ref<f32>
! CHECK:   %[[COMBINED:.*]] = arith.addf %[[LOAD0]], %[[LOAD1]] fastmath<contract> : f32
! CHECK:   fir.store %[[COMBINED]] to %[[ARG0]] : !fir.ref<f32>
! CHECK:   acc.yield %[[ARG0]] : !fir.ref<f32>
! CHECK: }

! CHECK-LABEL: acc.reduction.recipe @reduction_add_section_ext100xext10xext2_ref_100x10x2xi32 : !fir.ref<!fir.array<100x10x2xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10x2xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}}, %{{.*}} : (index, index, index) -> !fir.shape<3>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10x2xi32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10x2xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<100x10x2xi32>>, !fir.ref<!fir.array<100x10x2xi32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100x10x2xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_add_section_ext100xext10_ref_100x10xi32 : !fir.ref<!fir.array<100x10xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}}, %{{.*}} : (index, index) -> !fir.shape<2>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100x10xi32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x10xi32>>, !fir.ref<!fir.array<100x10xi32>>)
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100x10xi32>>
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

! CHECK-LABEL: acc.reduction.recipe @reduction_add_section_ext100_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <add> init {
! CHECK: ^bb0(%{{.*}}: !fir.ref<!fir.array<100xi32>>):
! CHECK:   %[[INIT:.*]] = arith.constant 0 : i32
! CHECK:   %[[SHAPE:.*]] = fir.shape %{{.*}} : (index) -> !fir.shape<1>
! CHECK:   %[[ALLOCA:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]](%[[SHAPE]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! HFLIR:   acc.yield %[[DECLARE]]#0 : !fir.ref<!fir.array<100xi32>>
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
! CHECK:   %[[DECLARE:.*]]:2 = hlfir.declare %[[ALLOCA]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[INIT]] to %[[DECLARE]]#0 : !fir.ref<i32>
! CHECK:   acc.yield %[[DECLARE]]#0 : !fir.ref<i32>
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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_section_ext100_ref_100xi32 -> %[[RED_B]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK:       %[[DECLARG1:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:       %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10xi32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<100x10xi32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_section_ext100xext10_ref_100x10xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xi32>>)
! CHECK: } attributes {collapse = [2]{{.*}}

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
! CHECK: %[[DECLARG1:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10x2xi32>>) bounds(%{{.*}}, %{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<100x10x2xi32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_add_section_ext100xext10xext2_ref_100x10x2xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10x2xi32>>)
! CHECK: } attributes {collapse = [3]{{.*}}

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_section_ext100_ref_100xf32 -> %[[RED_B]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_mul_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_mul_section_ext100_ref_100xi32 -> %[[RED_B]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_mul_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_mul_section_ext100_ref_100xf32 -> %[[RED_B]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_min_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK: %[[DECLARG1:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xi32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_min_section_ext100_ref_100xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_min_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK: %[[DECLARG1:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<100x10xf32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_min_section_ext100xext10_ref_100x10xf32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xf32>>)
! CHECK: attributes {collapse = [2]{{.*}}

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<i32>) -> !fir.ref<i32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_max_ref_i32 -> %[[RED_B]] : !fir.ref<i32>)

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
! CHECK: %[[DECLARG1:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10xi32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<100x10xi32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_max_section_ext100xext10_ref_100x10xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xi32>>)

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
! CHECK:       %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<f32>) -> !fir.ref<f32> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_max_ref_f32 -> %[[RED_B]] : !fir.ref<f32>)

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
! CHECK:       %[[DECLARG1:.*]]:2 = hlfir.declare %[[ARG1]]
! CHECK:       %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_max_section_ext100_ref_100xf32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK: %[[L:.*]] = fir.alloca !fir.logical<4> {bindc_name = "l", uniq_name = "_QFacc_reduction_andEl"}
! CHECK: %[[DECLL:.*]]:2 = hlfir.declare %[[L]]
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[DECLL]]#0 : !fir.ref<!fir.logical<4>>) -> !fir.ref<!fir.logical<4>> {name = "l"}
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
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<complex<f32>>) -> !fir.ref<complex<f32>> {name = "c"}
! CHECK: acc.parallel reduction(@reduction_add_ref_z32 -> %[[RED]] : !fir.ref<complex<f32>>)

subroutine acc_reduction_mul_cmplx()
  complex :: c
  !$acc parallel reduction(*:c)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_mul_cmplx()
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<complex<f32>>) -> !fir.ref<complex<f32>> {name = "c"}
! CHECK: acc.parallel reduction(@reduction_mul_ref_z32 -> %[[RED]] : !fir.ref<complex<f32>>)

subroutine acc_reduction_add_alloc()
  integer, allocatable :: i
  allocate(i)
  !$acc parallel reduction(+:i)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_alloc()
! CHECK: %[[ALLOCA:.*]] = fir.alloca !fir.box<!fir.heap<i32>> {bindc_name = "i", uniq_name = "_QFacc_reduction_add_allocEi"}
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ALLOCA]]
! CHECK: %[[LOAD:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
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
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[LOAD:.*]] = fir.load %[[DECLARG0]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
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
! CHECK: %[[C100:.*]] = arith.constant 100 : index
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 10 : index
! CHECK: %[[UB:.*]] = arith.constant 19 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%[[C100]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[DECLARG0]]#0 : !fir.ref<!fir.array<100xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<100xi32>> {name = "a(11:20)"}
! CHECK: acc.parallel reduction(@reduction_add_section_lb10.ub19_ref_100xi32 -> %[[RED]] : !fir.ref<!fir.array<100xi32>>)

subroutine acc_reduction_add_dynamic_extent_add(a)
  integer :: a(:)
  !$acc parallel reduction(+:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_dynamic_extent_add(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"})
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<!fir.array<?xi32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xi32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_add_box_Uxi32 -> %[[RED:.*]] : !fir.ref<!fir.array<?xi32>>)

subroutine acc_reduction_add_assumed_shape_max(a)
  real :: a(:)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_assumed_shape_max(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"})
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%{{.*}} : !fir.ref<!fir.array<?xf32>>) bounds(%{{.*}}) -> !fir.ref<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_box_Uxf32 -> %[[RED]] : !fir.ref<!fir.array<?xf32>>) {

subroutine acc_reduction_add_dynamic_extent_add_with_section(a)
  integer :: a(:)
  !$acc parallel reduction(+:a(2:4))
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_dynamic_extent_add_with_section(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {uniq_name = "_QFacc_reduction_add_dynamic_extent_add_with_sectionEa"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c1{{.*}} : index) upperbound(%c3{{.*}} : index) extent(%{{.*}}#1 : index) stride(%{{.*}}#2 : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[DECL]]#0 : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[BOX_ADDR]] : !fir.ref<!fir.array<?xi32>>) bounds(%[[BOUND]]) -> !fir.ref<!fir.array<?xi32>> {name = "a(2:4)"}
! CHECK: acc.parallel reduction(@reduction_add_section_lb1.ub3_box_Uxi32 -> %[[RED]] : !fir.ref<!fir.array<?xi32>>)

subroutine acc_reduction_add_allocatable(a)
  real, allocatable :: a(:)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_allocatable(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFacc_reduction_add_allocatableEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}}#1 : index) stride(%{{.*}}#2 : index) startIdx(%{{.*}}#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[BOX_ADDR]] : !fir.heap<!fir.array<?xf32>>)   bounds(%{{[0-9]+}}) -> !fir.heap<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_box_heap_Uxf32 -> %[[RED]] : !fir.heap<!fir.array<?xf32>>)

subroutine acc_reduction_add_pointer_array(a)
  real, pointer :: a(:)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_pointer_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFacc_reduction_add_pointer_arrayEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK: %[[BOX:.*]] = fir.load %[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c0{{.*}} : index) upperbound(%{{.*}} : index) extent(%{{.*}}#1 : index) stride(%{{.*}}#2 : index) startIdx(%{{.*}}#0 : index) {strideInBytes = true}
! CHECK: %[[BOX_ADDR:.*]] = fir.box_addr %[[BOX]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[BOX_ADDR]] : !fir.ptr<!fir.array<?xf32>>) bounds(%[[BOUND]]) -> !fir.ptr<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_box_ptr_Uxf32 -> %[[RED]] : !fir.ptr<!fir.array<?xf32>>)

subroutine acc_reduction_max_dynamic_extent_max(a, n)
  integer :: n
  real :: a(n, n)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_max_dynamic_extent_max(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "a"}, %{{.*}}: !fir.ref<i32> {fir.bindc_name = "n"})
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} {uniq_name = "_QFacc_reduction_max_dynamic_extent_maxEa"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
! CHECK: %[[ADDR:.*]] = fir.box_addr %[[DECL_A]]#0 : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[ADDR]] : !fir.ref<!fir.array<?x?xf32>>) bounds(%{{.*}}, %{{.*}}) -> !fir.ref<!fir.array<?x?xf32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_box_UxUxf32 -> %[[RED]] : !fir.ref<!fir.array<?x?xf32>>)
