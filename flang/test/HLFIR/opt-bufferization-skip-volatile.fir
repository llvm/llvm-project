// RUN: fir-opt --pass-pipeline="builtin.module(func.func(opt-bufferization))" %s | FileCheck %s

// Ensure optimized bufferization preserves the semantics of volatile arrays
func.func @minimal_volatile_test() {
  %c1 = arith.constant 1 : index
  %c200 = arith.constant 200 : index

  // Create a volatile array
  %1 = fir.address_of(@_QMtestEarray) : !fir.ref<!fir.array<200xf32>>
  %2 = fir.shape %c200 : (index) -> !fir.shape<1>
  %3 = fir.volatile_cast %1 : (!fir.ref<!fir.array<200xf32>>) -> !fir.ref<!fir.array<200xf32>, volatile>
  %4:2 = hlfir.declare %3(%2) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QMtestEarray"} : (!fir.ref<!fir.array<200xf32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>, volatile>, !fir.ref<!fir.array<200xf32>, volatile>)

  // Create an elemental operation that negates each element
  %5 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<200xf32> {
  ^bb0(%arg1: index):
    %6 = hlfir.designate %4#0 (%arg1) : (!fir.ref<!fir.array<200xf32>, volatile>, index) -> !fir.ref<f32, volatile>
    %7 = fir.load %6 : !fir.ref<f32, volatile>
    %8 = arith.negf %7 : f32
    hlfir.yield_element %8 : f32
  }

  // Assign the result back to the volatile array
  hlfir.assign %5 to %4#0 : !hlfir.expr<200xf32>, !fir.ref<!fir.array<200xf32>, volatile>
  hlfir.destroy %5 : !hlfir.expr<200xf32>

  return
}

fir.global @_QMtestEarray : !fir.array<200xf32> 

// CHECK-LABEL:   func.func @minimal_volatile_test() {
// CHECK:           %[[VAL_0:.*]] = arith.constant 200 : index
// CHECK:           %[[VAL_1:.*]] = fir.address_of(@_QMtestEarray) : !fir.ref<!fir.array<200xf32>>
// CHECK:           %[[VAL_2:.*]] = fir.shape %[[VAL_0]] : (index) -> !fir.shape<1>
// CHECK:           %[[VAL_3:.*]] = fir.volatile_cast %[[VAL_1]] : (!fir.ref<!fir.array<200xf32>>) -> !fir.ref<!fir.array<200xf32>, volatile>
// CHECK:           %[[VAL_4:.*]]:2 = hlfir.declare %[[VAL_3]](%[[VAL_2]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QMtestEarray"} : (!fir.ref<!fir.array<200xf32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>, volatile>, !fir.ref<!fir.array<200xf32>, volatile>)
// CHECK:           %[[VAL_5:.*]] = hlfir.elemental %[[VAL_2]] unordered : (!fir.shape<1>) -> !hlfir.expr<200xf32> {
// CHECK:           ^bb0(%[[VAL_6:.*]]: index):
// CHECK:             %[[VAL_7:.*]] = hlfir.designate %[[VAL_4]]#0 (%[[VAL_6]])  : (!fir.ref<!fir.array<200xf32>, volatile>, index) -> !fir.ref<f32, volatile>
// CHECK:             %[[VAL_8:.*]] = fir.load %[[VAL_7]] : !fir.ref<f32, volatile>
// CHECK:             %[[VAL_9:.*]] = arith.negf %[[VAL_8]] : f32
// CHECK:             hlfir.yield_element %[[VAL_9]] : f32
// CHECK:           }
// CHECK:           hlfir.assign %[[VAL_5]] to %[[VAL_4]]#0 : !hlfir.expr<200xf32>, !fir.ref<!fir.array<200xf32>, volatile>
// CHECK:           hlfir.destroy %[[VAL_5]] : !hlfir.expr<200xf32>
// CHECK:           return
// CHECK:         }
// CHECK:         fir.global @_QMtestEarray : !fir.array<200xf32>

fir.global @_QMtestEarray_src : !fir.array<200xf32>
fir.global @_QMtestEarray_dst : !fir.array<200xf32>

// Check that 'dst = -volatile_src' is optimized.
func.func @volatile_src_nonvolatile_dst() {
  %c1 = arith.constant 1 : index
  %c200 = arith.constant 200 : index

  %1 = fir.address_of(@_QMtestEarray_src) : !fir.ref<!fir.array<200xf32>>
  %dst = fir.address_of(@_QMtestEarray_dst) : !fir.ref<!fir.array<200xf32>>
  %2 = fir.shape %c200 : (index) -> !fir.shape<1>
  %3 = fir.volatile_cast %1 : (!fir.ref<!fir.array<200xf32>>) -> !fir.ref<!fir.array<200xf32>, volatile>
  %4:2 = hlfir.declare %3(%2) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QMtestEarray_src"} : (!fir.ref<!fir.array<200xf32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>, volatile>, !fir.ref<!fir.array<200xf32>, volatile>)
  %dstDecl:2 = hlfir.declare %dst(%2) {uniq_name = "_QMtestEarray_dst"} : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>>, !fir.ref<!fir.array<200xf32>>)

  %5 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<200xf32> {
  ^bb0(%arg1: index):
    %6 = hlfir.designate %4#0 (%arg1) : (!fir.ref<!fir.array<200xf32>, volatile>, index) -> !fir.ref<f32, volatile>
    %7 = fir.load %6 : !fir.ref<f32, volatile>
    %8 = arith.negf %7 : f32
    hlfir.yield_element %8 : f32
  }

  hlfir.assign %5 to %dstDecl#0 : !hlfir.expr<200xf32>, !fir.ref<!fir.array<200xf32>>
  hlfir.destroy %5 : !hlfir.expr<200xf32>

  return
}

// CHECK-LABEL:   func.func @volatile_src_nonvolatile_dst() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 200 : index
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtestEarray_src) : !fir.ref<!fir.array<200xf32>>
// CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QMtestEarray_dst) : !fir.ref<!fir.array<200xf32>>
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ADDRESS_OF_0]] : (!fir.ref<!fir.array<200xf32>>) -> !fir.ref<!fir.array<200xf32>, volatile>
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QMtestEarray_src"} : (!fir.ref<!fir.array<200xf32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>, volatile>, !fir.ref<!fir.array<200xf32>, volatile>)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ADDRESS_OF_1]](%[[SHAPE_0]]) {uniq_name = "_QMtestEarray_dst"} : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>>, !fir.ref<!fir.array<200xf32>>)
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_0]] unordered {
// CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[VAL_0]])  : (!fir.ref<!fir.array<200xf32>, volatile>, index) -> !fir.ref<f32, volatile>
// CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32, volatile>
// CHECK:             %[[NEGF_0:.*]] = arith.negf %[[LOAD_0]] : f32
// CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[DECLARE_1]]#0 (%[[VAL_0]])  : (!fir.ref<!fir.array<200xf32>>, index) -> !fir.ref<f32>
// CHECK:             hlfir.assign %[[NEGF_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32>
// CHECK:           }
// CHECK:           return
// CHECK:         }

// Check that 'volatile_dst = -src' is optimized.
func.func @nonvolatile_src_volatile_dst() {
  %c1 = arith.constant 1 : index
  %c200 = arith.constant 200 : index

  %1 = fir.address_of(@_QMtestEarray_dst) : !fir.ref<!fir.array<200xf32>>
  %dst = fir.address_of(@_QMtestEarray_src) : !fir.ref<!fir.array<200xf32>>
  %2 = fir.shape %c200 : (index) -> !fir.shape<1>
  %3 = fir.volatile_cast %1 : (!fir.ref<!fir.array<200xf32>>) -> !fir.ref<!fir.array<200xf32>, volatile>
  %4:2 = hlfir.declare %3(%2) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QMtestEarray_dst"} : (!fir.ref<!fir.array<200xf32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>, volatile>, !fir.ref<!fir.array<200xf32>, volatile>)
  %srcDecl:2 = hlfir.declare %dst(%2) {uniq_name = "_QMtestEarray_src"} : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>>, !fir.ref<!fir.array<200xf32>>)

  %5 = hlfir.elemental %2 unordered : (!fir.shape<1>) -> !hlfir.expr<200xf32> {
  ^bb0(%arg1: index):
    %6 = hlfir.designate %srcDecl#0 (%arg1) : (!fir.ref<!fir.array<200xf32>>, index) -> !fir.ref<f32>
    %7 = fir.load %6 : !fir.ref<f32>
    %8 = arith.negf %7 : f32
    hlfir.yield_element %8 : f32
  }

  hlfir.assign %5 to %4#0 : !hlfir.expr<200xf32>, !fir.ref<!fir.array<200xf32>, volatile>
  hlfir.destroy %5 : !hlfir.expr<200xf32>

  return
}
// CHECK-LABEL:   func.func @nonvolatile_src_volatile_dst() {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 200 : index
// CHECK:           %[[ADDRESS_OF_0:.*]] = fir.address_of(@_QMtestEarray_dst) : !fir.ref<!fir.array<200xf32>>
// CHECK:           %[[ADDRESS_OF_1:.*]] = fir.address_of(@_QMtestEarray_src) : !fir.ref<!fir.array<200xf32>>
// CHECK:           %[[SHAPE_0:.*]] = fir.shape %[[CONSTANT_1]] : (index) -> !fir.shape<1>
// CHECK:           %[[VOLATILE_CAST_0:.*]] = fir.volatile_cast %[[ADDRESS_OF_0]] : (!fir.ref<!fir.array<200xf32>>) -> !fir.ref<!fir.array<200xf32>, volatile>
// CHECK:           %[[DECLARE_0:.*]]:2 = hlfir.declare %[[VOLATILE_CAST_0]](%[[SHAPE_0]]) {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "_QMtestEarray_dst"} : (!fir.ref<!fir.array<200xf32>, volatile>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>, volatile>, !fir.ref<!fir.array<200xf32>, volatile>)
// CHECK:           %[[DECLARE_1:.*]]:2 = hlfir.declare %[[ADDRESS_OF_1]](%[[SHAPE_0]]) {uniq_name = "_QMtestEarray_src"} : (!fir.ref<!fir.array<200xf32>>, !fir.shape<1>) -> (!fir.ref<!fir.array<200xf32>>, !fir.ref<!fir.array<200xf32>>)
// CHECK:           fir.do_loop %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_0]] unordered {
// CHECK:             %[[DESIGNATE_0:.*]] = hlfir.designate %[[DECLARE_1]]#0 (%[[VAL_0]])  : (!fir.ref<!fir.array<200xf32>>, index) -> !fir.ref<f32>
// CHECK:             %[[LOAD_0:.*]] = fir.load %[[DESIGNATE_0]] : !fir.ref<f32>
// CHECK:             %[[NEGF_0:.*]] = arith.negf %[[LOAD_0]] : f32
// CHECK:             %[[DESIGNATE_1:.*]] = hlfir.designate %[[DECLARE_0]]#0 (%[[VAL_0]])  : (!fir.ref<!fir.array<200xf32>, volatile>, index) -> !fir.ref<f32, volatile>
// CHECK:             hlfir.assign %[[NEGF_0]] to %[[DESIGNATE_1]] : f32, !fir.ref<f32, volatile>
// CHECK:           }
// CHECK:           return
// CHECK:         }
