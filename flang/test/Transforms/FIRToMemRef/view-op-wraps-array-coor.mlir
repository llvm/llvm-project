// Test that zero-offset FortranObjectViewOpInterface ops wrapping a
// fir.array_coor are peeled through, so the array_coor is still lowered to a
// bounds-aware memref access instead of being treated as an opaque scalar
// reference. fir.volatile_cast is a peel barrier.
// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s
// The pass must not introduce any fir.convert that drops volatility.
// RUN: fir-opt %s --strict-fir-volatile-verifier --fir-to-memref -o /dev/null

// CHECK-LABEL: func.func @declare_wraps_array_coor
// CHECK:       %[[M0:.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<10xi32>>) -> memref<10xi32>
// CHECK:       %[[M1:.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<10xi32>>) -> memref<10xi32>
// CHECK:       memref.load %[[M0]]
// CHECK:       memref.store %arg1, %[[M1]]
// CHECK-NOT:   fir.array_coor
func.func @declare_wraps_array_coor(%arg0: !fir.ref<!fir.array<10xi32>>, %v: i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %decl = fir.declare %elem {uniq_name = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %load = fir.load %decl : !fir.ref<i32>
  fir.store %v to %decl : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @convert_wraps_array_coor
// CHECK:       %[[M0:.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<10xi32>>) -> memref<10xi32>
// CHECK:       %[[M1:.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<10xi32>>) -> memref<10xi32>
// CHECK:       memref.load %[[M0]]
// CHECK:       memref.store %arg1, %[[M1]]
// CHECK-NOT:   fir.array_coor
func.func @convert_wraps_array_coor(%arg0: !fir.ref<!fir.array<10xi32>>, %v: i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %cvt = fir.convert %elem : (!fir.ref<i32>) -> !fir.ref<i32>
  %load = fir.load %cvt : !fir.ref<i32>
  fir.store %v to %cvt : !fir.ref<i32>
  return
}

// A fir.box_addr can only reach an array_coor through an intervening
// fir.embox (its own operand is always a box, and array_coor never produces
// one), so this also exercises peeling through fir.embox.
// CHECK-LABEL: func.func @box_addr_wraps_array_coor
// CHECK:       %[[M0:.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<10xi32>>) -> memref<10xi32>
// CHECK:       %[[M1:.+]] = fir.convert %arg0 : (!fir.ref<!fir.array<10xi32>>) -> memref<10xi32>
// CHECK:       memref.load %[[M0]]
// CHECK:       memref.store %arg1, %[[M1]]
// CHECK-NOT:   fir.array_coor
func.func @box_addr_wraps_array_coor(%arg0: !fir.ref<!fir.array<10xi32>>, %v: i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %box = fir.embox %elem : (!fir.ref<i32>) -> !fir.box<i32>
  %addr = fir.box_addr %box : (!fir.box<i32>) -> !fir.ref<i32>
  %load = fir.load %addr : !fir.ref<i32>
  fir.store %v to %addr : !fir.ref<i32>
  return
}

// A fir.volatile_cast can go from volatile to non-volatile, e.g. after taking
// the address of an element of a volatile array. Unlike the other view ops,
// peeling must *not* look past it: the fir.array_coor underneath is still
// volatile, so converting its base directly to memref would bypass the cast
// and silently drop volatility. Instead only the (already non-volatile)
// fir.volatile_cast result itself is marshaled, as a rank-0 memref, leaving
// fir.array_coor untouched in FIR.
// CHECK-LABEL: func.func @volatile_cast_wraps_array_coor
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<i32, volatile>
// CHECK:       %[[VC:.+]] = fir.volatile_cast %[[COOR]] : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
// CHECK:       %[[M0:.+]] = fir.convert %[[VC]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:       memref.load %[[M0]][]
// CHECK:       %[[M1:.+]] = fir.convert %[[VC]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:       memref.store %arg1, %[[M1]][]
func.func @volatile_cast_wraps_array_coor(%arg0: !fir.ref<!fir.array<10xi32>, volatile>, %v: i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<10xi32>, volatile>, !fir.shape<1>, index) -> !fir.ref<i32, volatile>
  %vc = fir.volatile_cast %elem : (!fir.ref<i32, volatile>) -> !fir.ref<i32>
  %load = fir.load %vc : !fir.ref<i32>
  fir.store %v to %vc : !fir.ref<i32>
  return
}

// Negative case: fir.volatile_cast going the other way, non-volatile to
// volatile (the common direction, e.g. declaring a variable volatile). Here
// the load/store's own type ends up volatile, so the pass must bail out
// before ever reaching getMemRefInfo/peeling — nothing should be converted.
// CHECK-LABEL: func.func @volatile_cast_to_volatile_wraps_array_coor
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<i32>
// CHECK:       %[[VC:.+]] = fir.volatile_cast %[[COOR]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:       fir.load %[[VC]] : !fir.ref<i32, volatile>
// CHECK:       fir.store %arg1 to %[[VC]] : !fir.ref<i32, volatile>
// CHECK-NOT:   memref
func.func @volatile_cast_to_volatile_wraps_array_coor(%arg0: !fir.ref<!fir.array<10xi32>>, %v: i32) {
  %c1 = arith.constant 1 : index
  %c10 = arith.constant 10 : index
  %shape = fir.shape %c10 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<10xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %vc = fir.volatile_cast %elem : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %load = fir.load %vc : !fir.ref<i32, volatile>
  fir.store %v to %vc : !fir.ref<i32, volatile>
  return
}
