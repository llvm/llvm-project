! This test checks lowering of OpenACC reduction clause.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_box_UxUxf32 : !fir.box<!fir.array<?x?xf32>> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?x?xf32>, %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCMEM_0]](%[[SHAPE_0]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?x?xf32>>, !fir.shape<2>) -> (!fir.box<!fir.array<?x?xf32>>, !fir.heap<!fir.array<?x?xf32>>)
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : f32, !fir.box<!fir.array<?x?xf32>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.box<!fir.array<?x?xf32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_2]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_3:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_3]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_2]]#1, %[[BOX_DIMS_3]]#1 : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[BOX_DIMS_1]]#1 step %[[CONSTANT_4]] unordered {
! CHECK:             fir.do_loop %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[BOX_DIMS_0]]#1 step %[[CONSTANT_4]] unordered {
! CHECK:               %[[CONSTANT_5:.*]] = arith.constant 0 : index
! CHECK:               %[[BOX_DIMS_4:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_5]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:               %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:               %[[BOX_DIMS_5:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_6]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:               %[[CONSTANT_7:.*]] = arith.constant 1 : index
! CHECK:               %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_4]]#0, %[[CONSTANT_7]] : index
! CHECK:               %[[ADDI_0:.*]] = arith.addi %[[VAL_3]], %[[SUBI_0]] : index
! CHECK:               %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_5]]#0, %[[CONSTANT_7]] : index
! CHECK:               %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[SUBI_1]] : index
! CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[ADDI_0]], %[[ADDI_1]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
! CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:               %[[CONSTANT_8:.*]] = arith.constant 0 : index
! CHECK:               %[[BOX_DIMS_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_8]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:               %[[CONSTANT_9:.*]] = arith.constant 1 : index
! CHECK:               %[[BOX_DIMS_7:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_9]] : (!fir.box<!fir.array<?x?xf32>>, index) -> (index, index, index)
! CHECK:               %[[CONSTANT_10:.*]] = arith.constant 1 : index
! CHECK:               %[[SUBI_2:.*]] = arith.subi %[[BOX_DIMS_6]]#0, %[[CONSTANT_10]] : index
! CHECK:               %[[ADDI_2:.*]] = arith.addi %[[VAL_3]], %[[SUBI_2]] : index
! CHECK:               %[[SUBI_3:.*]] = arith.subi %[[BOX_DIMS_7]]#0, %[[CONSTANT_10]] : index
! CHECK:               %[[ADDI_3:.*]] = arith.addi %[[VAL_2]], %[[SUBI_3]] : index
! CHECK:               %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[ADDI_2]], %[[ADDI_3]])  : (!fir.box<!fir.array<?x?xf32>>, index, index) -> !fir.ref<f32>
! CHECK:               %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:               %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32
! CHECK:               hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.box<!fir.array<?x?xf32>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?xf32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?xf32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?xf32>>) -> !fir.ref<!fir.array<?x?xf32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?x?xf32>>) -> !fir.heap<!fir.array<?x?xf32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?x?xf32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_ref_box_ptr_Uxf32 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xf32>, %[[BOX_DIMS_0]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCMEM_0]](%[[SHAPE_0]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xf32>>>
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_1]]#0 : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:           fir.store %[[DECLARE_0]]#0 to %[[CONVERT_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_1]]#0 : f32, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           acc.yield %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_1]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_1]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[BOX_DIMS_0]]#1 step %[[CONSTANT_2]] unordered {
! CHECK:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_3]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_2]]#0, %[[CONSTANT_4]] : index
! CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[SUBI_0]] : index
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[LOAD_0]] (%[[ADDI_0]])  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_2:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_3:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_5]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_3]]#0, %[[CONSTANT_6]] : index
! CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[SUBI_1]] : index
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[LOAD_1]] (%[[ADDI_1]])  : (!fir.box<!fir.ptr<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_3:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_3]], %[[LOAD_2]] fastmath<contract> : f32
! CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_3]], %[[LOAD_2]] : f32
! CHECK:             hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.ptr<!fir.array<?xf32>>>) -> !fir.ptr<!fir.array<?xf32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ptr<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_ref_box_heap_Uxf32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_1]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xf32>, %[[BOX_DIMS_0]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCMEM_0]](%[[SHAPE_0]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xf32>>>
! CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[DECLARE_1]]#0 : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:           fir.store %[[DECLARE_0]]#0 to %[[CONVERT_0]] : !fir.ref<!fir.box<!fir.array<?xf32>>>
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_1]]#0 : f32, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           acc.yield %[[DECLARE_1]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_1]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_1]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[BOX_DIMS_0]]#1 step %[[CONSTANT_2]] unordered {
! CHECK:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_3]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_2]]#0, %[[CONSTANT_4]] : index
! CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[SUBI_0]] : index
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[LOAD_0]] (%[[ADDI_0]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_2:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_3:.*]]:3 = fir.box_dims %[[LOAD_1]], %[[CONSTANT_5]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_3]]#0, %[[CONSTANT_6]] : index
! CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[SUBI_1]] : index
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[LOAD_1]] (%[[ADDI_1]])  : (!fir.box<!fir.heap<!fir.array<?xf32>>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_3:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_3]], %[[LOAD_2]] fastmath<contract> : f32
! CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_3]], %[[LOAD_2]] : f32
! CHECK:             hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.heap<!fir.array<?xf32>>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_section_lb1.ub3_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS_0]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCMEM_0]](%[[SHAPE_0]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.box<!fir.array<?xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 3 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_1]], %[[CONSTANT_0]] : index
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SUBI_0]], %[[CONSTANT_2]] : index
! CHECK:           %[[DIVSI_0:.*]] = arith.divsi %[[ADDI_0]], %[[CONSTANT_2]] : index
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[DIVSI_0]], %[[CONSTANT_3]] : index
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[DIVSI_0]], %[[CONSTANT_3]] : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[SELECT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[BD_LHS:.*]]:3 = fir.box_dims %[[VAL_0]], %c0{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[LB_LHS:.*]] = arith.addi %[[BD_LHS]]#0, %c1{{.*}} : index
! CHECK:           %[[UB_LHS:.*]] = arith.addi %[[BD_LHS]]#0, %c3{{.*}} : index
! CHECK:           %[[BD_RHS:.*]]:3 = fir.box_dims %[[VAL_1]], %c0{{.*}} : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[LB_RHS:.*]] = arith.addi %[[BD_RHS]]#0, %c1{{.*}} : index
! CHECK:           %[[UB_RHS:.*]] = arith.addi %[[BD_RHS]]#0, %c3{{.*}} : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[LB_RHS]]:%[[UB_RHS]]:%c1{{.*}})  shape %[[SHAPE_0]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[LB_LHS]]:%[[UB_LHS]]:%c1{{.*}})  shape %[[SHAPE_0]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[SELECT_0]] step %[[CONSTANT_4]] unordered {
! CHECK:             %[[DESIGNATE_2:.*]] = hlfir.designate %[[DESIGNATE_0]] (%[[VAL_2]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_2]] : !fir.ref<i32>
! CHECK:             %[[DESIGNATE_3:.*]] = hlfir.designate %[[DESIGNATE_1]] (%[[VAL_2]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_3]] : !fir.ref<i32>
! CHECK:             %[[ADDI_1:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:             hlfir.assign %[[ADDI_1]] to %[[DESIGNATE_3]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.box<!fir.array<?xi32>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_box_Uxf32 : !fir.box<!fir.array<?xf32>> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xf32>, %[[BOX_DIMS_0]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCMEM_0]](%[[SHAPE_0]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xf32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xf32>>, !fir.heap<!fir.array<?xf32>>)
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : f32, !fir.box<!fir.array<?xf32>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.box<!fir.array<?xf32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_1]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[BOX_DIMS_0]]#1 step %[[CONSTANT_2]] unordered {
! CHECK:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_3]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_2]]#0, %[[CONSTANT_4]] : index
! CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[SUBI_0]] : index
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[ADDI_0]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_5]] : (!fir.box<!fir.array<?xf32>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_3]]#0, %[[CONSTANT_6]] : index
! CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[SUBI_1]] : index
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[ADDI_1]])  : (!fir.box<!fir.array<?xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32
! CHECK:             hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.box<!fir.array<?xf32>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xf32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xf32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xf32>>) -> !fir.ref<!fir.array<?xf32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?xf32>>) -> !fir.heap<!fir.array<?xf32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?xf32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_box_Uxi32 : !fir.box<!fir.array<?xi32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS_0]]#1 {bindc_name = ".tmp", uniq_name = ""}
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCMEM_0]](%[[SHAPE_0]]) {uniq_name = ".tmp"} : (!fir.heap<!fir.array<?xi32>>, !fir.shape<1>) -> (!fir.box<!fir.array<?xi32>>, !fir.heap<!fir.array<?xi32>>)
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : i32, !fir.box<!fir.array<?xi32>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.box<!fir.array<?xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_1]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[BOX_DIMS_0]]#1 step %[[CONSTANT_2]] unordered {
! CHECK:             %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_3]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_0:.*]] = arith.subi %[[BOX_DIMS_2]]#0, %[[CONSTANT_4]] : index
! CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_2]], %[[SUBI_0]] : index
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[ADDI_0]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
! CHECK:             %[[CONSTANT_5:.*]] = arith.constant 0 : index
! CHECK:             %[[BOX_DIMS_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_5]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:             %[[SUBI_1:.*]] = arith.subi %[[BOX_DIMS_3]]#0, %[[CONSTANT_6]] : index
! CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[SUBI_1]] : index
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[ADDI_1]])  : (!fir.box<!fir.array<?xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<i32>
! CHECK:             %[[ADDI_2:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:             hlfir.assign %[[ADDI_2]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.box<!fir.array<?xi32>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_section_lb0.ub9xlb0.ub19_ref_10x20xi32 : !fir.ref<!fir.array<10x20xi32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<10x20xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 20 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]], %[[CONSTANT_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<10x20xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<10x20xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<10x20xi32>>, !fir.ref<!fir.array<10x20xi32>>)
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 19 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_5]] {
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 0 : index
! CHECK:             %[[CONSTANT_7:.*]] = arith.constant 9 : index
! CHECK:             %[[CONSTANT_8:.*]] = arith.constant 1 : index
! CHECK:             fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] {
! CHECK:               %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_2]], %[[VAL_1]] : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<10x20xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<10x20xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<10x20xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 9 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 19 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant 0 : index
! CHECK:           %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_1]], %[[CONSTANT_0]] : index
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SUBI_0]], %[[CONSTANT_2]] : index
! CHECK:           %[[DIVSI_0:.*]] = arith.divsi %[[ADDI_0]], %[[CONSTANT_2]] : index
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[DIVSI_0]], %[[CONSTANT_6]] : index
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[DIVSI_0]], %[[CONSTANT_6]] : index
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 0 : index
! CHECK:           %[[SUBI_1:.*]] = arith.subi %[[CONSTANT_4]], %[[CONSTANT_3]] : index
! CHECK:           %[[ADDI_1:.*]] = arith.addi %[[SUBI_1]], %[[CONSTANT_5]] : index
! CHECK:           %[[DIVSI_1:.*]] = arith.divsi %[[ADDI_1]], %[[CONSTANT_5]] : index
! CHECK:           %[[CMPI_1:.*]] = arith.cmpi sgt, %[[DIVSI_1]], %[[CONSTANT_7]] : index
! CHECK:           %[[SELECT_1:.*]] = arith.select %[[CMPI_1]], %[[DIVSI_1]], %[[CONSTANT_7]] : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[SELECT_0]], %[[SELECT_1]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%c1{{.*}}:%c10{{.*}}:%c1{{.*}}, %c1{{.*}}:%c20{{.*}}:%c1{{.*}})  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<10x20xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<10x20xi32>>
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%c1{{.*}}:%c10{{.*}}:%c1{{.*}}, %c1{{.*}}:%c20{{.*}}:%c1{{.*}})  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<10x20xi32>>, index, index, index, index, index, index, !fir.shape<2>) -> !fir.ref<!fir.array<10x20xi32>>
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_8]] to %[[SELECT_1]] step %[[CONSTANT_8]] unordered {
! CHECK:             fir.do_loop %[[VAL_3:.*]] = %[[CONSTANT_8]] to %[[SELECT_0]] step %[[CONSTANT_8]] unordered {
! CHECK:               %[[DESIGNATE_2:.*]] = hlfir.designate %[[DESIGNATE_0]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_2]] : !fir.ref<i32>
! CHECK:               %[[DESIGNATE_3:.*]] = hlfir.designate %[[DESIGNATE_1]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<10x20xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_3]] : !fir.ref<i32>
! CHECK:               %[[ADDI_2:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:               hlfir.assign %[[ADDI_2]] to %[[DESIGNATE_3]] : i32, !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<10x20xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_section_lb10.ub19_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
! CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_1]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 10 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 19 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[SUBI_0:.*]] = arith.subi %[[CONSTANT_1]], %[[CONSTANT_0]] : index
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[SUBI_0]], %[[CONSTANT_2]] : index
! CHECK:           %[[DIVSI_0:.*]] = arith.divsi %[[ADDI_0]], %[[CONSTANT_2]] : index
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[DIVSI_0]], %[[CONSTANT_3]] : index
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[DIVSI_0]], %[[CONSTANT_3]] : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[SELECT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%c11{{.*}}:%c20{{.*}}:%c1{{.*}})  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<100xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%c11{{.*}}:%c20{{.*}}:%c1{{.*}})  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<100xi32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<100xi32>>
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[SELECT_0]] step %[[CONSTANT_4]] unordered {
! CHECK:             %[[DESIGNATE_2:.*]] = hlfir.designate %[[DESIGNATE_0]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_2]] : !fir.ref<i32>
! CHECK:             %[[DESIGNATE_3:.*]] = hlfir.designate %[[DESIGNATE_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_3]] : !fir.ref<i32>
! CHECK:             %[[ADDI_1:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:             hlfir.assign %[[ADDI_1]] to %[[DESIGNATE_3]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_box_ptr_i32 : !fir.ref<!fir.box<!fir.ptr<i32>>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.ptr<i32>>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.box<!fir.ptr<i32>>>) -> (!fir.ref<!fir.box<!fir.ptr<i32>>>, !fir.ref<!fir.box<!fir.ptr<i32>>>)
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem i32
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]] : (!fir.heap<i32>) -> !fir.box<!fir.ptr<i32>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[DECLARE_0]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : i32, !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[BOX_ADDR_1:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.ptr<i32>
! CHECK:           %[[LOAD_3:.*]] = fir.load %[[BOX_ADDR_1]] : !fir.ptr<i32>
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[LOAD_2]] : i32
! CHECK:           hlfir.assign %[[ADDI_0]] to %[[BOX_ADDR_1]] : i32, !fir.ptr<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<i32>>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<i32>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.ptr<i32>>) -> !fir.ptr<i32>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ptr<i32>) -> !fir.heap<i32>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<i32>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_box_heap_i32 : !fir.ref<!fir.box<!fir.heap<i32>>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.box<!fir.heap<i32>>>) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem i32
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[DECLARE_0]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           hlfir.assign %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : i32, !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[BOX_ADDR_1:.*]] = fir.box_addr %[[LOAD_1]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[LOAD_2:.*]] = fir.load %[[BOX_ADDR_0]] : !fir.heap<i32>
! CHECK:           %[[LOAD_3:.*]] = fir.load %[[BOX_ADDR_1]] : !fir.heap<i32>
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_3]], %[[LOAD_2]] : i32
! CHECK:           hlfir.assign %[[ADDI_0]] to %[[BOX_ADDR_1]] : i32, !fir.heap<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<i32>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_mul_ref_z32 : !fir.ref<complex<f32>> reduction_operator <mul> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<complex<f32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined complex<f32>
! CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_0]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[CONSTANT_1]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca complex<f32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK:           fir.store %[[INSERT_VALUE_1]] to %[[DECLARE_0]]#0 : !fir.ref<complex<f32>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<complex<f32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<complex<f32>>, %[[VAL_1:.*]]: !fir.ref<complex<f32>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<complex<f32>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<complex<f32>>
! CHECK:           %[[MULC_0:.*]] = fir.mulc %[[LOAD_1]], %[[LOAD_0]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:           hlfir.assign %[[MULC_0]] to %[[VAL_0]] : complex<f32>, !fir.ref<complex<f32>>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<complex<f32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_z32 : !fir.ref<complex<f32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<complex<f32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[UNDEFINED_0:.*]] = fir.undefined complex<f32>
! CHECK:           %[[INSERT_VALUE_0:.*]] = fir.insert_value %[[UNDEFINED_0]], %[[CONSTANT_0]], [0 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:           %[[INSERT_VALUE_1:.*]] = fir.insert_value %[[INSERT_VALUE_0]], %[[CONSTANT_1]], [1 : index] : (complex<f32>, f32) -> complex<f32>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca complex<f32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<complex<f32>>) -> (!fir.ref<complex<f32>>, !fir.ref<complex<f32>>)
! CHECK:           fir.store %[[INSERT_VALUE_1]] to %[[DECLARE_0]]#0 : !fir.ref<complex<f32>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<complex<f32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<complex<f32>>, %[[VAL_1:.*]]: !fir.ref<complex<f32>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<complex<f32>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<complex<f32>>
! CHECK:           %[[ADDC_0:.*]] = fir.addc %[[LOAD_1]], %[[LOAD_0]] {fastmath = #arith.fastmath<contract>} : complex<f32>
! CHECK:           hlfir.assign %[[ADDC_0]] to %[[VAL_0]] : complex<f32>, !fir.ref<complex<f32>>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<complex<f32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_neqv_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <neqv> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant false
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.logical<4>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[CONSTANT_0]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[CONVERT_0]] to %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>, %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_1]] : (!fir.logical<4>) -> i1
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[LOAD_0]] : (!fir.logical<4>) -> i1
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi ne, %[[CONVERT_0]], %[[CONVERT_1]] : i1
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[CMPI_0]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[CONVERT_2]] to %[[VAL_0]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_eqv_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <eqv> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant true
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.logical<4>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[CONSTANT_0]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[CONVERT_0]] to %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>, %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_1]] : (!fir.logical<4>) -> i1
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[LOAD_0]] : (!fir.logical<4>) -> i1
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[CONVERT_0]], %[[CONVERT_1]] : i1
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[CMPI_0]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[CONVERT_2]] to %[[VAL_0]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_lor_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <lor> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant false
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.logical<4>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[CONSTANT_0]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[CONVERT_0]] to %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>, %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_1]] : (!fir.logical<4>) -> i1
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[LOAD_0]] : (!fir.logical<4>) -> i1
! CHECK:           %[[ORI_0:.*]] = arith.ori %[[CONVERT_0]], %[[CONVERT_1]] : i1
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[ORI_0]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[CONVERT_2]] to %[[VAL_0]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_land_ref_l32 : !fir.ref<!fir.logical<4>> reduction_operator <land> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant true
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.logical<4>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.logical<4>>) -> (!fir.ref<!fir.logical<4>>, !fir.ref<!fir.logical<4>>)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[CONSTANT_0]] : (i1) -> !fir.logical<4>
! CHECK:           fir.store %[[CONVERT_0]] to %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.logical<4>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.logical<4>>, %[[VAL_1:.*]]: !fir.ref<!fir.logical<4>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[LOAD_1]] : (!fir.logical<4>) -> i1
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[LOAD_0]] : (!fir.logical<4>) -> i1
! CHECK:           %[[ANDI_0:.*]] = arith.andi %[[CONVERT_0]], %[[CONVERT_1]] : i1
! CHECK:           %[[CONVERT_2:.*]] = fir.convert %[[ANDI_0]] : (i1) -> !fir.logical<4>
! CHECK:           hlfir.assign %[[CONVERT_2]] to %[[VAL_0]] : !fir.logical<4>, !fir.ref<!fir.logical<4>>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.logical<4>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_xor_ref_i32 : !fir.ref<i32> reduction_operator <xor> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[XORI_0:.*]] = arith.xori %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           hlfir.assign %[[XORI_0]] to %[[VAL_0]] : i32, !fir.ref<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<i32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_ior_ref_i32 : !fir.ref<i32> reduction_operator <ior> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[ORI_0:.*]] = arith.ori %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           hlfir.assign %[[ORI_0]] to %[[VAL_0]] : i32, !fir.ref<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<i32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_iand_ref_i32 : !fir.ref<i32> reduction_operator <iand> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[ANDI_0:.*]] = arith.andi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           hlfir.assign %[[ANDI_0]] to %[[VAL_0]] : i32, !fir.ref<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<i32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
! CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_1]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xf32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_2]] unordered {
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:             %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32
! CHECK:             hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100xf32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_ref_f32 : !fir.ref<f32> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -1.401300e-45 : f32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca f32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<f32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<f32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>, %[[VAL_1:.*]]: !fir.ref<f32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<f32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK:           %[[CMPF_0:.*]] = arith.cmpf ogt, %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32
! CHECK:           hlfir.assign %[[SELECT_0]] to %[[VAL_0]] : f32, !fir.ref<f32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<f32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_ref_100x10xi32 : !fir.ref<!fir.array<100x10xi32>> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -2147483648 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]], %[[CONSTANT_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100x10xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x10xi32>>, !fir.ref<!fir.array<100x10xi32>>)
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 9 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_5]] {
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 0 : index
! CHECK:             %[[CONSTANT_7:.*]] = arith.constant 99 : index
! CHECK:             %[[CONSTANT_8:.*]] = arith.constant 1 : index
! CHECK:             fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] {
! CHECK:               %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_2]], %[[VAL_1]] : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100x10xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]], %[[CONSTANT_1]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_2]], %[[CONSTANT_3]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_1]] step %[[CONSTANT_4]] unordered {
! CHECK:             fir.do_loop %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_0]] step %[[CONSTANT_4]] unordered {
! CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
! CHECK:               %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<i32>
! CHECK:               %[[CMPI_0:.*]] = arith.cmpi sgt, %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:               hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100x10xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_max_ref_i32 : !fir.ref<i32> reduction_operator <max> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant -2147483648 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi sgt, %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           hlfir.assign %[[SELECT_0]] to %[[VAL_0]] : i32, !fir.ref<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<i32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_min_ref_100x10xf32 : !fir.ref<!fir.array<100x10xf32>> reduction_operator <min> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3.40282347E+38 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]], %[[CONSTANT_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100x10xf32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10xf32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x10xf32>>, !fir.ref<!fir.array<100x10xf32>>)
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 9 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_5]] {
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 0 : index
! CHECK:             %[[CONSTANT_7:.*]] = arith.constant 99 : index
! CHECK:             %[[CONSTANT_8:.*]] = arith.constant 1 : index
! CHECK:             fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] {
! CHECK:               %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_2]], %[[VAL_1]] : (!fir.ref<!fir.array<100x10xf32>>, index, index) -> !fir.ref<f32>
! CHECK:               fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100x10xf32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100x10xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]], %[[CONSTANT_1]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_2]], %[[CONSTANT_3]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_1]] step %[[CONSTANT_4]] unordered {
! CHECK:             fir.do_loop %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_0]] step %[[CONSTANT_4]] unordered {
! CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10xf32>>, index, index) -> !fir.ref<f32>
! CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:               %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10xf32>>, index, index) -> !fir.ref<f32>
! CHECK:               %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:               %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:               %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32
! CHECK:               hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100x10xf32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_min_ref_f32 : !fir.ref<f32> reduction_operator <min> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 3.40282347E+38 : f32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca f32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<f32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<f32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>, %[[VAL_1:.*]]: !fir.ref<f32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<f32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK:           %[[CMPF_0:.*]] = arith.cmpf olt, %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPF_0]], %[[LOAD_1]], %[[LOAD_0]] : f32
! CHECK:           hlfir.assign %[[SELECT_0]] to %[[VAL_0]] : f32, !fir.ref<f32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<f32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_min_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <min> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2147483647 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
! CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_1]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_2]] unordered {
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<i32>
! CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:             %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:             hlfir.assign %[[SELECT_0]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_min_ref_i32 : !fir.ref<i32> reduction_operator <min> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2147483647 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[CMPI_0:.*]] = arith.cmpi slt, %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           hlfir.assign %[[SELECT_0]] to %[[VAL_0]] : i32, !fir.ref<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<i32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_mul_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <mul> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
! CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_1]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xf32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_2]] unordered {
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:             %[[MULF_0:.*]] = arith.mulf %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:             hlfir.assign %[[MULF_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100xf32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_mul_ref_f32 : !fir.ref<f32> reduction_operator <mul> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1.000000e+00 : f32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca f32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<f32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<f32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>, %[[VAL_1:.*]]: !fir.ref<f32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<f32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK:           %[[MULF_0:.*]] = arith.mulf %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:           hlfir.assign %[[MULF_0]] to %[[VAL_0]] : f32, !fir.ref<f32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<f32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_mul_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <mul> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
! CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_1]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_2]] unordered {
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<i32>
! CHECK:             %[[MULI_0:.*]] = arith.muli %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:             hlfir.assign %[[MULI_0]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_mul_ref_i32 : !fir.ref<i32> reduction_operator <mul> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[MULI_0:.*]] = arith.muli %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           hlfir.assign %[[MULI_0]] to %[[VAL_0]] : i32, !fir.ref<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<i32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_100xf32 : !fir.ref<!fir.array<100xf32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xf32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
! CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_1]] : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xf32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_2]] unordered {
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xf32>>, index) -> !fir.ref<f32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<f32>
! CHECK:             %[[ADDF_0:.*]] = arith.addf %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:             hlfir.assign %[[ADDF_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100xf32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_f32 : !fir.ref<f32> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca f32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<f32>) -> (!fir.ref<f32>, !fir.ref<f32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<f32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<f32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<f32>, %[[VAL_1:.*]]: !fir.ref<f32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<f32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<f32>
! CHECK:           %[[ADDF_0:.*]] = arith.addf %[[LOAD_1]], %[[LOAD_0]] fastmath<contract> : f32
! CHECK:           hlfir.assign %[[ADDF_0]] to %[[VAL_0]] : f32, !fir.ref<f32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<f32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_100x10x2xi32 : !fir.ref<!fir.array<100x10x2xi32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10x2xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]], %[[CONSTANT_2]], %[[CONSTANT_3]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100x10x2xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10x2xi32>>, !fir.shape<3>) -> (!fir.ref<!fir.array<100x10x2xi32>>, !fir.ref<!fir.array<100x10x2xi32>>)
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_5]] step %[[CONSTANT_6]] {
! CHECK:             %[[CONSTANT_7:.*]] = arith.constant 0 : index
! CHECK:             %[[CONSTANT_8:.*]] = arith.constant 9 : index
! CHECK:             %[[CONSTANT_9:.*]] = arith.constant 1 : index
! CHECK:             fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_7]] to %[[CONSTANT_8]] step %[[CONSTANT_9]] {
! CHECK:               %[[CONSTANT_10:.*]] = arith.constant 0 : index
! CHECK:               %[[CONSTANT_11:.*]] = arith.constant 99 : index
! CHECK:               %[[CONSTANT_12:.*]] = arith.constant 1 : index
! CHECK:               fir.do_loop %[[VAL_3:.*]] = %[[CONSTANT_10]] to %[[CONSTANT_11]] step %[[CONSTANT_12]] {
! CHECK:                 %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_3]], %[[VAL_2]], %[[VAL_1]] : (!fir.ref<!fir.array<100x10x2xi32>>, index, index, index) -> !fir.ref<i32>
! CHECK:                 fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100x10x2xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10x2xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100x10x2xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]], %[[CONSTANT_1]], %[[CONSTANT_2]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 10 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_3]], %[[CONSTANT_4]], %[[CONSTANT_5]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_2]] step %[[CONSTANT_6]] unordered {
! CHECK:             fir.do_loop %[[VAL_3:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_1]] step %[[CONSTANT_6]] unordered {
! CHECK:               fir.do_loop %[[VAL_4:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_0]] step %[[CONSTANT_6]] unordered {
! CHECK:                 %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_4]], %[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10x2xi32>>, index, index, index) -> !fir.ref<i32>
! CHECK:                 %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
! CHECK:                 %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_4]], %[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10x2xi32>>, index, index, index) -> !fir.ref<i32>
! CHECK:                 %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<i32>
! CHECK:                 %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:                 hlfir.assign %[[ADDI_0]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32>
! CHECK:               }
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100x10x2xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_100x10xi32 : !fir.ref<!fir.array<100x10xi32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]], %[[CONSTANT_2]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100x10xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100x10xi32>>, !fir.shape<2>) -> (!fir.ref<!fir.array<100x10xi32>>, !fir.ref<!fir.array<100x10xi32>>)
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 9 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_3]] to %[[CONSTANT_4]] step %[[CONSTANT_5]] {
! CHECK:             %[[CONSTANT_6:.*]] = arith.constant 0 : index
! CHECK:             %[[CONSTANT_7:.*]] = arith.constant 99 : index
! CHECK:             %[[CONSTANT_8:.*]] = arith.constant 1 : index
! CHECK:             fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] {
! CHECK:               %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_2]], %[[VAL_1]] : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100x10xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100x10xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100x10xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]], %[[CONSTANT_1]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 100 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_2]], %[[CONSTANT_3]] : (index, index) -> !fir.shape<2>
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_1]] step %[[CONSTANT_4]] unordered {
! CHECK:             fir.do_loop %[[VAL_3:.*]] = %[[CONSTANT_4]] to %[[CONSTANT_0]] step %[[CONSTANT_4]] unordered {
! CHECK:               %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
! CHECK:               %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_3]], %[[VAL_2]])  : (!fir.ref<!fir.array<100x10xi32>>, index, index) -> !fir.ref<i32>
! CHECK:               %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<i32>
! CHECK:               %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:               hlfir.assign %[[ADDI_0]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32>
! CHECK:             }
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100x10xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_100xi32 : !fir.ref<!fir.array<100xi32>> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xi32>
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]](%[[SHAPE_0]]) {uniq_name = "acc.reduction.init"} : (!fir.ref<!fir.array<100xi32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<100xi32>>, !fir.ref<!fir.array<100xi32>>)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_1:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_4]] {
! CHECK:             %[[COORDINATE_OF_0:.*]] = fir.coordinate_of %[[DECLARE_0]]#0, %[[VAL_1]] : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             fir.store %[[CONSTANT_0]] to %[[COORDINATE_OF_0]] : !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<!fir.array<100xi32>>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xi32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : index
! CHECK:           fir.do_loop %[[VAL_2:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_0]] step %[[CONSTANT_2]] unordered {
! CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_1]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<i32>
! CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_0]] (%[[VAL_2]])  : (!fir.ref<!fir.array<100xi32>>, index) -> !fir.ref<i32>
! CHECK:             %[[LOAD_1:.*]] = fir.load %[[DESIGNATE_1]] : !fir.ref<i32>
! CHECK:             %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:             hlfir.assign %[[ADDI_0]] to %[[DESIGNATE_1]] : i32, !fir.ref<i32>
! CHECK:           }
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<!fir.array<100xi32>>
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.private.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:         }

! CHECK-LABEL:   acc.reduction.recipe @reduction_add_ref_i32 : !fir.ref<i32> reduction_operator <add> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32
! CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[ALLOCA_0]] {uniq_name = "acc.reduction.init"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:           fir.store %[[CONSTANT_0]] to %[[DECLARE_0]]#0 : !fir.ref<i32>
! CHECK:           acc.yield %[[DECLARE_0]]#0 : !fir.ref<i32>

! CHECK-LABEL:   } combiner {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<i32>
! CHECK:           %[[LOAD_1:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[LOAD_1]], %[[LOAD_0]] : i32
! CHECK:           hlfir.assign %[[ADDI_0]] to %[[VAL_0]] : i32, !fir.ref<i32>
! CHECK:           acc.yield %[[VAL_0]] : !fir.ref<i32>
! CHECK:         }

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_ref_100xi32 -> %[[RED_B]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK:       %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10xi32>>) -> !fir.ref<!fir.array<100x10xi32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_ref_100x10xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xi32>>)
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
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10x2xi32>>) -> !fir.ref<!fir.array<100x10x2xi32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_add_ref_100x10x2xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10x2xi32>>)
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
! CHECK: %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_add_ref_100xf32 -> %[[RED_B]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_mul_ref_100xi32 -> %[[RED_B]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK:       %[[RED_B:.*]] = acc.reduction varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_mul_ref_100xf32 -> %[[RED_B]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100xi32>>) -> !fir.ref<!fir.array<100xi32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_min_ref_100xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100xi32>>)

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
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10xf32>>) -> !fir.ref<!fir.array<100x10xf32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_min_ref_100x10xf32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xf32>>)
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
! CHECK: %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100x10xi32>>) -> !fir.ref<!fir.array<100x10xi32>> {name = "b"}
! CHECK: acc.loop {{.*}} reduction(@reduction_max_ref_100x10xi32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100x10xi32>>)

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
! CHECK:       %[[RED_ARG1:.*]] = acc.reduction varPtr(%[[DECLARG1]]#0 : !fir.ref<!fir.array<100xf32>>) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK:       acc.loop {{.*}} reduction(@reduction_max_ref_100xf32 -> %[[RED_ARG1]] : !fir.ref<!fir.array<100xf32>>)

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
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>) -> !fir.ref<!fir.box<!fir.heap<i32>>> {name = "i"}
! CHECK: acc.parallel reduction(@reduction_add_ref_box_heap_i32 -> %[[RED]] : !fir.ref<!fir.box<!fir.heap<i32>>>)

subroutine acc_reduction_add_pointer(i)
  integer, pointer :: i
  !$acc parallel reduction(+:i)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_pointer(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<i32>>> {fir.bindc_name = "i"})
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[DECLARG0]]#0 : !fir.ref<!fir.box<!fir.ptr<i32>>>) -> !fir.ref<!fir.box<!fir.ptr<i32>>> {name = "i"}
! CHECK: acc.parallel reduction(@reduction_add_ref_box_ptr_i32 -> %[[RED]] : !fir.ref<!fir.box<!fir.ptr<i32>>>)

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

subroutine acc_reduction_add_static_slice_2d(a)
  integer :: a(10,20)
  !$acc parallel reduction(+:a(:10,:20))
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_static_slice_2d(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<10x20xi32>> {fir.bindc_name = "a"})
! CHECK: %[[C10:.*]] = arith.constant 10 : index
! CHECK: %[[C20:.*]] = arith.constant 20 : index
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[UB9:.*]] = arith.constant 9 : index
! CHECK: %[[STRIDE1:.*]] = arith.constant 10 : index
! CHECK: %[[BOUND0:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB9]] : index) extent(%[[C10]] : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[UB19:.*]] = arith.constant 19 : index
! CHECK: %[[BOUND1:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB19]] : index) extent(%[[C20]] : index)
! stride(%[[STRIDE1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[DECLARG0]]#0 : !fir.ref<!fir.array<10x20xi32>>) bounds(%[[BOUND0]], %[[BOUND1]]) ->
! !fir.ref<!fir.array<10x20xi32>> {name = "a(:10,:20)"}
! CHECK: acc.parallel reduction(@reduction_add_section_lb0.ub9xlb0.ub19_ref_10x20xi32 -> %[[RED]] : !fir.ref<!fir.array<10x20xi32>>)

subroutine acc_reduction_add_dynamic_extent_add(a)
  integer :: a(:)
  !$acc parallel reduction(+:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_dynamic_extent_add(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"})
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[RED:.*]] = acc.reduction var(%{{.*}} : !fir.box<!fir.array<?xi32>>) -> !fir.box<!fir.array<?xi32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_add_box_Uxi32 -> %[[RED:.*]] : !fir.box<!fir.array<?xi32>>)

subroutine acc_reduction_add_assumed_shape_max(a)
  real :: a(:)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_assumed_shape_max(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xf32>> {fir.bindc_name = "a"})
! CHECK: %[[DECLARG0:.*]]:2 = hlfir.declare %[[ARG0]]
! CHECK: %[[RED:.*]] = acc.reduction var(%{{.*}} : !fir.box<!fir.array<?xf32>>) -> !fir.box<!fir.array<?xf32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_box_Uxf32 -> %[[RED]] : !fir.box<!fir.array<?xf32>>) {

subroutine acc_reduction_add_dynamic_extent_add_with_section(a)
  integer :: a(:)
  !$acc parallel reduction(+:a(2:4))
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_dynamic_extent_add_with_section(
! CHECK-SAME: %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFacc_reduction_add_dynamic_extent_add_with_sectionEa"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%c1{{.*}} : index) upperbound(%c3{{.*}} : index) extent(%{{.*}}#1 : index) stride(%{{.*}}#2 : index) startIdx(%{{.*}} : index) {strideInBytes = true}
! CHECK: %[[RED:.*]] = acc.reduction var(%[[DECL]]#0 : !fir.box<!fir.array<?xi32>>) bounds(%[[BOUND]]) -> !fir.box<!fir.array<?xi32>> {name = "a(2:4)"}
! CHECK: acc.parallel reduction(@reduction_add_section_lb1.ub3_box_Uxi32 -> %[[RED]] : !fir.box<!fir.array<?xi32>>)

subroutine acc_reduction_add_allocatable(a)
  real, allocatable :: a(:)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_allocatable(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFacc_reduction_add_allocatableEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[DECL]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_ref_box_heap_Uxf32 -> %[[RED]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xf32>>>>)

subroutine acc_reduction_add_pointer_array(a)
  real, pointer :: a(:)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_add_pointer_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {fir.bindc_name = "a"})
! CHECK: %[[DECL:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFacc_reduction_add_pointer_arrayEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)
! CHECK: %[[RED:.*]] = acc.reduction varPtr(%[[DECL]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_ref_box_ptr_Uxf32 -> %[[RED]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xf32>>>>)

subroutine acc_reduction_max_dynamic_extent_max(a, n)
  integer :: n
  real :: a(n, n)
  !$acc parallel reduction(max:a)
  !$acc end parallel
end subroutine

! CHECK-LABEL: func.func @_QPacc_reduction_max_dynamic_extent_max(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?x?xf32>> {fir.bindc_name = "a"}, %{{.*}}: !fir.ref<i32> {fir.bindc_name = "n"})
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFacc_reduction_max_dynamic_extent_maxEa"} : (!fir.ref<!fir.array<?x?xf32>>, !fir.shape<2>, !fir.dscope) -> (!fir.box<!fir.array<?x?xf32>>, !fir.ref<!fir.array<?x?xf32>>)
! CHECK: %[[RED:.*]] = acc.reduction var(%[[DECL_A]]#0 : !fir.box<!fir.array<?x?xf32>>) -> !fir.box<!fir.array<?x?xf32>> {name = "a"}
! CHECK: acc.parallel reduction(@reduction_max_box_UxUxf32 -> %[[RED]] : !fir.box<!fir.array<?x?xf32>>)
