! This test checks lowering of OpenACC loop directive.

! RUN: bbc -fopenacc -emit-hlfir %s -o - | FileCheck %s


! CHECK-LABEL:   acc.private.recipe @privatization_ptr_10xf32 : !fir.ptr<!fir.array<10xf32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ptr<!fir.array<10xf32>>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<10xf32> {bindc_name = "acc.private.init"}
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 10 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : index
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<!fir.array<10xf32>>
! CHECK:         }

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_box_UxUx2xi32 : !fir.box<!fir.array<?x?x2xi32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?x2xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1, %[[CONSTANT_2]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1, %[[CONSTANT_2]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?x?x2xi32>, %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1 {bindc_name = "acc.private.init", uniq_name = ""}
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_3]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_4]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_5]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_6]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 2 : index
! CHECK:           %[[BOX_DIMS_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_7]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_2]]#0, %[[BOX_DIMS_3]]#1, %[[BOX_DIMS_4]]#0, %[[BOX_DIMS_5]]#1, %[[BOX_DIMS_6]]#0, %[[CONSTANT_8]] : (index, index, index, index, index, index) -> !fir.shapeshift<3>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]](%[[SHAPE_SHIFT_0]]) : (!fir.heap<!fir.array<?x?x2xi32>>, !fir.shapeshift<3>) -> !fir.box<!fir.array<?x?x2xi32>>
! CHECK:           acc.yield %[[EMBOX_0]] : !fir.box<!fir.array<?x?x2xi32>>

! CHECK-LABEL:   } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?x2xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x2xi32>>):
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_1]] temporary_lhs : !fir.box<!fir.array<?x?x2xi32>>, !fir.box<!fir.array<?x?x2xi32>>
! CHECK:           acc.terminator

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?x2xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x2xi32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x2xi32>>) -> !fir.ref<!fir.array<?x?x2xi32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?x?x2xi32>>) -> !fir.heap<!fir.array<?x?x2xi32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?x?x2xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_section_lb4.ub9_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<6xi32> {bindc_name = "acc.private.init"}
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 9 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 5 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 6 : index
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant true
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_7]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[BOX_DIMS_0]]#0, %[[CONSTANT_1]] : index
! CHECK:           %[[ADDI_1:.*]] = arith.addi %[[BOX_DIMS_0]]#0, %[[CONSTANT_2]] : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_0]] (%[[ADDI_0]]:%[[ADDI_1]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<6xi32>>
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_8]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_9:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_9]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<!fir.array<6xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[CONSTANT_10:.*]] = arith.constant 0 : index
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[CONSTANT_1]], %[[BOX_DIMS_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[CONVERT_0]](%[[SHAPE_SHIFT_0]]) %[[CONSTANT_10]] : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>, index) -> !fir.ref<i32>
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[ARRAY_COOR_0]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[SHAPE_SHIFT_1:.*]] = fir.shape_shift %[[BOX_DIMS_1]]#0, %[[BOX_DIMS_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[CONVERT_1]](%[[SHAPE_SHIFT_1]]) : (!fir.ref<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           acc.yield %[[EMBOX_0]] : !fir.box<!fir.array<?xi32>>

! CHECK-LABEL:   } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 4 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 9 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 5 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 6 : index
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant true
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_1]], %[[CONSTANT_7]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[ADDI_0:.*]] = arith.addi %[[BOX_DIMS_0]]#0, %[[CONSTANT_1]] : index
! CHECK:           %[[ADDI_1:.*]] = arith.addi %[[BOX_DIMS_0]]#0, %[[CONSTANT_2]] : index
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_8]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[ADDI_2:.*]] = arith.addi %[[BOX_DIMS_1]]#0, %[[CONSTANT_1]] : index
! CHECK:           %[[ADDI_3:.*]] = arith.addi %[[BOX_DIMS_1]]#0, %[[CONSTANT_2]] : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_0]] (%[[ADDI_2]]:%[[ADDI_3]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<6xi32>>
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_1]] (%[[ADDI_0]]:%[[ADDI_1]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.box<!fir.array<?xi32>>, index, index, index, !fir.shape<1>) -> !fir.box<!fir.array<6xi32>>
! CHECK:           hlfir.assign %[[DESIGNATE_0]] to %[[DESIGNATE_1]] temporary_lhs : !fir.box<!fir.array<6xi32>>, !fir.box<!fir.array<6xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS_0]]#1 {bindc_name = "acc.private.init", uniq_name = ""}
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_2]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_1]]#0, %[[BOX_DIMS_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]](%[[SHAPE_SHIFT_0]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           acc.yield %[[EMBOX_0]] : !fir.box<!fir.array<?xi32>>

! CHECK-LABEL:   } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_1]] temporary_lhs : !fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>
! CHECK:           acc.terminator

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_box_UxUx2xi32 : !fir.box<!fir.array<?x?x2xi32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?x2xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1, %[[CONSTANT_2]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1, %[[CONSTANT_2]] : (index, index, index) -> !fir.shape<3>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?x?x2xi32>, %[[BOX_DIMS_0]]#1, %[[BOX_DIMS_1]]#1 {bindc_name = "acc.private.init", uniq_name = ""}
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_3]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_3:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_4]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_4:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_5]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:           %[[BOX_DIMS_5:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_6]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 2 : index
! CHECK:           %[[BOX_DIMS_6:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_7]] : (!fir.box<!fir.array<?x?x2xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 2 : index
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_2]]#0, %[[BOX_DIMS_3]]#1, %[[BOX_DIMS_4]]#0, %[[BOX_DIMS_5]]#1, %[[BOX_DIMS_6]]#0, %[[CONSTANT_8]] : (index, index, index, index, index, index) -> !fir.shapeshift<3>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]](%[[SHAPE_SHIFT_0]]) : (!fir.heap<!fir.array<?x?x2xi32>>, !fir.shapeshift<3>) -> !fir.box<!fir.array<?x?x2xi32>>
! CHECK:           acc.yield %[[EMBOX_0]] : !fir.box<!fir.array<?x?x2xi32>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?x?x2xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?x?x2xi32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?x?x2xi32>>) -> !fir.ref<!fir.array<?x?x2xi32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?x?x2xi32>>) -> !fir.heap<!fir.array<?x?x2xi32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?x?x2xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_ref_box_ptr_Uxi32 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_0]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS_0]]#1 {bindc_name = "acc.private.init", uniq_name = ""}
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_1]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_2]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_1]]#0, %[[BOX_DIMS_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]](%[[SHAPE_SHIFT_0]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.ptr<!fir.array<?xi32>>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.ptr<!fir.array<?xi32>>>) -> !fir.ptr<!fir.array<?xi32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ptr<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_ref_box_heap_i32 : !fir.ref<!fir.box<!fir.heap<i32>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem i32 {bindc_name = "acc.private.init", uniq_name = ""}
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]] : (!fir.heap<i32>) -> !fir.box<!fir.heap<i32>>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.heap<i32>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.heap<i32>>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.heap<i32>>) -> !fir.heap<i32>
! CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<i32>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_ref_box_heap_Uxi32 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS_0]]#1 {bindc_name = "acc.private.init", uniq_name = ""}
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_1]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[LOAD_0]], %[[CONSTANT_2]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_1]]#0, %[[BOX_DIMS_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]](%[[SHAPE_SHIFT_0]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.box<!fir.heap<!fir.array<?xi32>>>
! CHECK:           fir.store %[[EMBOX_0]] to %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, %[[VAL_1:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_1]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[LOAD_0]] : (!fir.box<!fir.heap<!fir.array<?xi32>>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.freemem %[[BOX_ADDR_0]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_box_Uxi32 : !fir.box<!fir.array<?xi32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_0:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_0]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[BOX_DIMS_0]]#1 : (index) -> !fir.shape<1>
! CHECK:           %[[ALLOCMEM_0:.*]] = fir.allocmem !fir.array<?xi32>, %[[BOX_DIMS_0]]#1 {bindc_name = "acc.private.init", uniq_name = ""}
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_1:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_1]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : index
! CHECK:           %[[BOX_DIMS_2:.*]]:3 = fir.box_dims %[[VAL_0]], %[[CONSTANT_2]] : (!fir.box<!fir.array<?xi32>>, index) -> (index, index, index)
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[BOX_DIMS_1]]#0, %[[BOX_DIMS_2]]#1 : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[EMBOX_0:.*]] = fir.embox %[[ALLOCMEM_0]](%[[SHAPE_SHIFT_0]]) : (!fir.heap<!fir.array<?xi32>>, !fir.shapeshift<1>) -> !fir.box<!fir.array<?xi32>>
! CHECK:           acc.yield %[[EMBOX_0]] : !fir.box<!fir.array<?xi32>>

! CHECK-LABEL:   } destroy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.box<!fir.array<?xi32>>, %[[VAL_1:.*]]: !fir.box<!fir.array<?xi32>>):
! CHECK:           %[[BOX_ADDR_0:.*]] = fir.box_addr %[[VAL_1]] : (!fir.box<!fir.array<?xi32>>) -> !fir.ref<!fir.array<?xi32>>
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[BOX_ADDR_0]] : (!fir.ref<!fir.array<?xi32>>) -> !fir.heap<!fir.array<?xi32>>
! CHECK:           fir.freemem %[[CONVERT_0]] : !fir.heap<!fir.array<?xi32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_section_lb50.ub99_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<50xf32> {bindc_name = "acc.private.init"}
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 50 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 49 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 50 : index
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant true
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 51 : index
! CHECK:           %[[CONSTANT_9:.*]] = arith.constant 100 : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_0]] (%[[CONSTANT_8]]:%[[CONSTANT_9]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<100xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<50xf32>>
! CHECK:           %[[CONSTANT_10:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_11:.*]] = arith.constant 100 : index
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<!fir.array<50xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:           %[[CONSTANT_12:.*]] = arith.constant 0 : index
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[CONSTANT_1]], %[[CONSTANT_11]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[CONVERT_0]](%[[SHAPE_SHIFT_0]]) %[[CONSTANT_12]] : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[ARRAY_COOR_0]] : (!fir.ref<f32>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:           acc.yield %[[CONVERT_1]] : !fir.ref<!fir.array<100xf32>>

! CHECK-LABEL:   } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 50 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 99 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 49 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant 50 : index
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant true
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_5]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 51 : index
! CHECK:           %[[CONSTANT_9:.*]] = arith.constant 100 : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_0]] (%[[CONSTANT_8]]:%[[CONSTANT_9]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<100xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<50xf32>>
! CHECK:           %[[DESIGNATE_1:.*]] = hlfir.designate %[[VAL_1]] (%[[CONSTANT_8]]:%[[CONSTANT_9]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<100xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<50xf32>>
! CHECK:           hlfir.assign %[[DESIGNATE_0]] to %[[DESIGNATE_1]] temporary_lhs : !fir.ref<!fir.array<50xf32>>, !fir.ref<!fir.array<50xf32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "acc.private.init"}
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<!fir.array<100xf32>>

! CHECK-LABEL:   } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>, %[[VAL_1:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           hlfir.assign %[[VAL_0]] to %[[VAL_1]] temporary_lhs : !fir.ref<!fir.array<100xf32>>, !fir.ref<!fir.array<100xf32>>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.firstprivate.recipe @firstprivatization_ref_i32 : !fir.ref<i32> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "acc.private.init"}
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<i32>

! CHECK-LABEL:   } copy {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>, %[[VAL_1:.*]]: !fir.ref<i32>):
! CHECK:           %[[LOAD_0:.*]] = fir.load %[[VAL_0]] : !fir.ref<i32>
! CHECK:           hlfir.assign %[[LOAD_0]] to %[[VAL_1]] temporary_lhs : i32, !fir.ref<i32>
! CHECK:           acc.terminator
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_section_lb0.ub49_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<50xf32> {bindc_name = "acc.private.init"}
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_2:.*]] = arith.constant 49 : index
! CHECK:           %[[CONSTANT_3:.*]] = arith.constant 0 : index
! CHECK:           %[[CONSTANT_4:.*]] = arith.constant 50 : index
! CHECK:           %[[CONSTANT_5:.*]] = arith.constant true
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_4]] : (index) -> !fir.shape<1>
! CHECK:           %[[CONSTANT_6:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_7:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_8:.*]] = arith.constant 50 : index
! CHECK:           %[[DESIGNATE_0:.*]] = hlfir.designate %[[VAL_0]] (%[[CONSTANT_7]]:%[[CONSTANT_8]]:%[[CONSTANT_0]])  shape %[[SHAPE_0]] : (!fir.ref<!fir.array<100xf32>>, index, index, index, !fir.shape<1>) -> !fir.ref<!fir.array<50xf32>>
! CHECK:           %[[CONSTANT_9:.*]] = arith.constant 1 : index
! CHECK:           %[[CONSTANT_10:.*]] = arith.constant 100 : index
! CHECK:           %[[CONVERT_0:.*]] = fir.convert %[[ALLOCA_0]] : (!fir.ref<!fir.array<50xf32>>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:           %[[CONSTANT_11:.*]] = arith.constant 0 : index
! CHECK:           %[[SHAPE_SHIFT_0:.*]] = fir.shape_shift %[[CONSTANT_1]], %[[CONSTANT_10]] : (index, index) -> !fir.shapeshift<1>
! CHECK:           %[[ARRAY_COOR_0:.*]] = fir.array_coor %[[CONVERT_0]](%[[SHAPE_SHIFT_0]]) %[[CONSTANT_11]] : (!fir.ref<!fir.array<100xf32>>, !fir.shapeshift<1>, index) -> !fir.ref<f32>
! CHECK:           %[[CONVERT_1:.*]] = fir.convert %[[ARRAY_COOR_0]] : (!fir.ref<f32>) -> !fir.ref<!fir.array<100xf32>>
! CHECK:           acc.yield %[[CONVERT_1]] : !fir.ref<!fir.array<100xf32>>
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_ref_100xf32 : !fir.ref<!fir.array<100xf32>> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<!fir.array<100xf32>>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca !fir.array<100xf32> {bindc_name = "acc.private.init"}
! CHECK:           %[[CONSTANT_0:.*]] = arith.constant 100 : index
! CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           %[[SHAPE_1:.*]] = fir.shape %[[CONSTANT_0]] : (index) -> !fir.shape<1>
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<!fir.array<100xf32>>
! CHECK:         }

! CHECK-LABEL:   acc.private.recipe @privatization_ref_i32 : !fir.ref<i32> init {
! CHECK:         ^bb0(%[[VAL_0:.*]]: !fir.ref<i32>):
! CHECK:           %[[ALLOCA_0:.*]] = fir.alloca i32 {bindc_name = "acc.private.init"}
! CHECK:           acc.yield %[[ALLOCA_0]] : !fir.ref<i32>
! CHECK:         }

program acc_private
  integer :: i, c
  integer, parameter :: n = 100
  real, dimension(n) :: a, b

! CHECK: %[[B:.*]] = fir.address_of(@_QFEb) : !fir.ref<!fir.array<100xf32>>
! CHECK: %[[DECLB:.*]]:2 = hlfir.declare %[[B]]
! CHECK: %[[C:.*]] = fir.alloca i32 {bindc_name = "c", uniq_name = "_QFEc"}
! CHECK: %[[DECLC:.*]]:2 = hlfir.declare %[[C]]

  !$acc loop private(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C_PRIVATE:.*]] = acc.private varPtr(%[[DECLC]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {name = "c"}
! CHECK: acc.loop private(%[[C_PRIVATE]]{{.*}} : !fir.ref<i32>{{.*}})
! CHECK: acc.yield

  !$acc loop private(b)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) recipe(@privatization_ref_100xf32) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK: acc.loop private(%[[B_PRIVATE]]{{.*}} : !fir.ref<!fir.array<100xf32>>{{.*}})
! CHECK: acc.yield

  !$acc loop private(b(1:50))
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 0 : index
! CHECK: %[[UB:.*]] = arith.constant 49 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[B_PRIVATE:.*]] = acc.private varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) recipe(@privatization_section_lb0.ub49_ref_100xf32) -> !fir.ref<!fir.array<100xf32>> {name = "b(1:50)"}
! CHECK: acc.loop private(%[[B_PRIVATE]]{{.*}} : !fir.ref<!fir.array<100xf32>>{{.*}})

  !$acc parallel loop firstprivate(c)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[FP_C:.*]] = acc.firstprivate varPtr(%[[DECLC]]#0 : !fir.ref<i32>) recipe(@firstprivatization_ref_i32) -> !fir.ref<i32> {name = "c"}
! CHECK: acc.parallel {{.*}} firstprivate(%[[FP_C]] : !fir.ref<i32>)
! CHECK: acc.yield

  !$acc parallel loop firstprivate(b)
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) recipe(@firstprivatization_ref_100xf32) -> !fir.ref<!fir.array<100xf32>> {name = "b"}
! CHECK: acc.parallel {{.*}} firstprivate(%[[FP_B]] : !fir.ref<!fir.array<100xf32>>)
! CHECK: acc.yield

  !$acc parallel loop firstprivate(b(51:100))
  DO i = 1, n
    c = i
    a(i) = b(i) + c
  END DO

! CHECK: %[[C1:.*]] = arith.constant 1 : index
! CHECK: %[[LB:.*]] = arith.constant 50 : index
! CHECK: %[[UB:.*]] = arith.constant 99 : index
! CHECK: %[[BOUND:.*]] = acc.bounds lowerbound(%[[LB]] : index) upperbound(%[[UB]] : index) extent(%{{.*}} : index) stride(%[[C1]] : index) startIdx(%[[C1]] : index)
! CHECK: %[[FP_B:.*]] = acc.firstprivate varPtr(%[[DECLB]]#0 : !fir.ref<!fir.array<100xf32>>) bounds(%[[BOUND]]) recipe(@firstprivatization_section_lb50.ub99_ref_100xf32) -> !fir.ref<!fir.array<100xf32>> {name = "b(51:100)"}
! CHECK: acc.parallel {{.*}} firstprivate(%[[FP_B]] : !fir.ref<!fir.array<100xf32>>)

end program

subroutine acc_private_assumed_shape(a, n)
  integer :: a(:), i, n

  !$acc parallel loop private(a)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_assumed_shape(
! CHECK-SAME:    %[[ARG0:.*]]: !fir.box<!fir.array<?xi32>> {fir.bindc_name = "a"}
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFacc_private_assumed_shapeEa"} : (!fir.box<!fir.array<?xi32>>, !fir.dscope) -> (!fir.box<!fir.array<?xi32>>, !fir.box<!fir.array<?xi32>>)
! CHECK: acc.parallel {{.*}} {
! CHECK: %[[PRIVATE:.*]] = acc.private var(%[[DECL_A]]#0 : !fir.box<!fir.array<?xi32>>) recipe(@privatization_box_Uxi32) -> !fir.box<!fir.array<?xi32>> {name = "a"}
! CHECK: acc.loop {{.*}} private(%[[PRIVATE]]{{.*}} : !fir.box<!fir.array<?xi32>>{{.*}})

subroutine acc_private_allocatable_array(a, n)
  integer, allocatable :: a(:)
  integer :: i, n

  !$acc parallel loop private(a)
  do i = 1, n
    a(i) = i
  end do

  !$acc serial private(a)
  a(i) = 1
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_allocatable_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {fir.bindc_name = "a"}
! CHECK: %[[DECLA_A:.*]]:2 = hlfir.declare %[[ARG0]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFacc_private_allocatable_arrayEa"} : (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)
! CHECK: acc.parallel {{.*}} {
! CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[DECLA_A]]#0 : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) recipe(@privatization_ref_box_heap_Uxi32) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>> {name = "a"}
! CHECK: acc.loop {{.*}} private(%[[PRIVATE]]{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>{{.*}})
! CHECK: %[[PRIVATE_SERIAL:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>) recipe(@privatization_ref_box_heap_Uxi32) -> !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>
! CHECK: acc.serial private(%[[PRIVATE_SERIAL]] : !fir.ref<!fir.box<!fir.heap<!fir.array<?xi32>>>>)

subroutine acc_private_allocatable_scalar(b, a, n)
  integer :: a(n)
  integer, allocatable :: b
  integer :: i, n

  !$acc parallel loop private(b)
  do i = 1, n
    a(i) = b
  end do

  !$acc serial private(b)
  a(i) = b
  !$acc end serial
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_allocatable_scalar(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.heap<i32>>> {fir.bindc_name = "b"}
! CHECK: %[[DECLA_B:.*]]:2 = hlfir.declare %arg0 dummy_scope %0 {{.*}} {fortran_attrs = #fir.var_attrs<allocatable>, uniq_name = "_QFacc_private_allocatable_scalarEb"} : (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.heap<i32>>>, !fir.ref<!fir.box<!fir.heap<i32>>>)
! CHECK: acc.parallel {{.*}} {
! CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[DECLA_B]]#0 : !fir.ref<!fir.box<!fir.heap<i32>>>) recipe(@privatization_ref_box_heap_i32) -> !fir.ref<!fir.box<!fir.heap<i32>>> {name = "b"}
! CHECK: acc.loop {{.*}} private(%[[PRIVATE]]{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>{{.*}})
! CHECK: %[[PRIVATE_SERIAL:.*]] = acc.private varPtr(%{{.*}} : !fir.ref<!fir.box<!fir.heap<i32>>>) recipe(@privatization_ref_box_heap_i32) -> !fir.ref<!fir.box<!fir.heap<i32>>>
! CHECK: acc.serial private(%[[PRIVATE_SERIAL]] : !fir.ref<!fir.box<!fir.heap<i32>>>) {

subroutine acc_private_pointer_array(a, n)
  integer, pointer :: a(:)
  integer :: i, n

  !$acc parallel loop private(a)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_pointer_array(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {fir.bindc_name = "a"}, %arg1: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %arg0 dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {fortran_attrs = #fir.var_attrs<pointer>, uniq_name = "_QFacc_private_pointer_arrayEa"} : (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.dscope) -> (!fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>, !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>)
! CHECK: acc.parallel {{.*}} {
! CHECK: %[[PRIVATE:.*]] = acc.private varPtr(%[[DECLA_A]]#0 : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>) recipe(@privatization_ref_box_ptr_Uxi32) -> !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>> {name = "a"}
! CHECK: acc.loop {{.*}} private(%[[PRIVATE]]{{.*}} : !fir.ref<!fir.box<!fir.ptr<!fir.array<?xi32>>>>{{.*}})

subroutine acc_private_dynamic_extent(a, n)
  integer :: n, i
  integer :: a(n, n, 2)

  !$acc parallel loop private(a)
  do i = 1, n
    a(i, i, 1) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_private_dynamic_extent(
! CHECK-SAME: %[[ARG0:.*]]: !fir.ref<!fir.array<?x?x2xi32>> {fir.bindc_name = "a"}, %[[ARG1:.*]]: !fir.ref<i32> {fir.bindc_name = "n"}) {
! CHECK: %[[DECL_N:.*]]:2 = hlfir.declare %[[ARG1]] dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFacc_private_dynamic_extentEn"} : (!fir.ref<i32>, !fir.dscope) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: %[[DECL_A:.*]]:2 = hlfir.declare %[[ARG0]](%{{.*}}) dummy_scope %{{[0-9]+}} arg {{[0-9]+}} {uniq_name = "_QFacc_private_dynamic_extentEa"} : (!fir.ref<!fir.array<?x?x2xi32>>, !fir.shape<3>, !fir.dscope) -> (!fir.box<!fir.array<?x?x2xi32>>, !fir.ref<!fir.array<?x?x2xi32>>)
! CHECK: acc.parallel {{.*}} {
! CHECK: %[[PRIV:.*]] = acc.private var(%[[DECL_A]]#0 : !fir.box<!fir.array<?x?x2xi32>>) recipe(@privatization_box_UxUx2xi32) -> !fir.box<!fir.array<?x?x2xi32>> {name = "a"}
! CHECK: acc.loop {{.*}} private(%[[PRIV]]{{.*}} : !fir.box<!fir.array<?x?x2xi32>>{{.*}})

subroutine acc_firstprivate_assumed_shape(a, n)
  integer :: a(:), i, n

  !$acc parallel loop firstprivate(a)
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_firstprivate_assumed_shape
! CHECK: %[[FIRSTPRIVATE_A:.*]] = acc.firstprivate var(%{{.*}} : !fir.box<!fir.array<?xi32>>) recipe(@firstprivatization_box_Uxi32) -> !fir.box<!fir.array<?xi32>> {name = "a"}
! CHECK: acc.parallel {{.*}}firstprivate(%[[FIRSTPRIVATE_A]] : !fir.box<!fir.array<?xi32>>) {

subroutine acc_firstprivate_assumed_shape_with_section(a, n)
  integer :: a(:), i, n

  !$acc parallel loop firstprivate(a(5:10))
  do i = 1, n
    a(i) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_firstprivate_assumed_shape_with_section
! CHECK: %[[FIRSTPRIVATE_A:.*]] = acc.firstprivate var(%{{.*}} : !fir.box<!fir.array<?xi32>>) bounds(%{{.*}}) recipe(@firstprivatization_section_lb4.ub9_box_Uxi32) -> !fir.box<!fir.array<?xi32>> {name = "a(5:10)"}
! CHECK: acc.parallel {{.*}}firstprivate(%[[FIRSTPRIVATE_A]] : !fir.box<!fir.array<?xi32>>)

subroutine acc_firstprivate_dynamic_extent(a, n)
  integer :: n, i
  integer :: a(n, n, 2)

  !$acc parallel loop firstprivate(a)
  do i = 1, n
    a(i, i, 1) = i
  end do
end subroutine

! CHECK-LABEL: func.func @_QPacc_firstprivate_dynamic_extent
! CHECK: %[[FIRSTPRIVATE_A:.*]] = acc.firstprivate var(%{{.*}} : !fir.box<!fir.array<?x?x2xi32>>) recipe(@firstprivatization_box_UxUx2xi32) -> !fir.box<!fir.array<?x?x2xi32>> {name = "a"}
! CHECK: acc.parallel {{.*}}firstprivate(%[[FIRSTPRIVATE_A]] : !fir.box<!fir.array<?x?x2xi32>>)

module acc_declare_equivalent
  integer, parameter :: n = 10
  real :: v1(n)
  real :: v2(n)
  equivalence(v1(1), v2(1))
contains
  subroutine sub1()
    !$acc parallel private(v2)
    !$acc end parallel
  end subroutine
end module

! CHECK: %[[PRIVATE_V2:.*]] = acc.private varPtr(%{{.*}} : !fir.ptr<!fir.array<10xf32>>) recipe(@privatization_ptr_10xf32) -> !fir.ptr<!fir.array<10xf32>>
! CHECK: acc.parallel private(%[[PRIVATE_V2]] : !fir.ptr<!fir.array<10xf32>>)

subroutine acc_private_use()
  integer :: i, j

  !$acc parallel loop
  do i = 1, 10
    j = i
  end do
end

! CHECK-LABEL: func.func @_QPacc_private_use()
! CHECK: %[[I:.*]] = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFacc_private_useEi"}
! CHECK: %[[DECL_I:.*]]:2 = hlfir.declare %[[I]] {uniq_name = "_QFacc_private_useEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK: acc.parallel
! CHECK: %[[PRIV_I:.*]] = acc.private varPtr(%[[DECL_I]]#0 : !fir.ref<i32>) recipe(@privatization_ref_i32) -> !fir.ref<i32> {implicit = true, name = "i"}
! CHECK: acc.loop {{.*}} private(%[[PRIV_I]] : !fir.ref<i32>) control(%[[IV0:.*]] : i32) = (%c1{{.*}} : i32) to (%c10{{.*}} : i32) step (%c1{{.*}} : i32)
! CHECK:   %[[DECL_PRIV_I:.*]]:2 = hlfir.declare %[[PRIV_I]] {uniq_name = "_QFacc_private_useEi"} : (!fir.ref<i32>) -> (!fir.ref<i32>, !fir.ref<i32>)
! CHECK:   fir.store %[[IV0]] to %[[DECL_PRIV_I]]#0 : !fir.ref<i32>
! CHECK:   %{{.*}} = fir.load %[[DECL_PRIV_I]]#0 : !fir.ref<i32>
