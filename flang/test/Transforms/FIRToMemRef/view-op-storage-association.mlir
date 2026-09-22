// A fir.convert that changes the pointee type of an fir.array_coor is Fortran
// storage association, not a transparent view. FIRToMemRef must marshal the
// typed view (rank-0 memref of the access type) rather than indexing the parent
// array and converting the loaded/stored value.
// RUN: fir-opt %s --fir-to-memref | FileCheck %s

//===----------------------------------------------------------------------===//
// complex element passed as real dummy
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @load_store_complex_as_real
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<complex<f32>>
// CHECK:       %[[CVT:.+]] = fir.convert %[[COOR]] : (!fir.ref<complex<f32>>) -> !fir.ref<f32>
// CHECK:       %[[DECL:.+]] = fir.declare %[[CVT]]
// CHECK:       %[[M0:.+]] = fir.convert %[[DECL]] : (!fir.ref<f32>) -> memref<f32>
// CHECK:       memref.load %[[M0]][] : memref<f32>
// CHECK:       %[[M1:.+]] = fir.convert %[[DECL]] : (!fir.ref<f32>) -> memref<f32>
// CHECK:       memref.store %arg1, %[[M1]][] : memref<f32>
// CHECK-NOT:   memref.load {{.*}} : memref<{{.*}}complex
// CHECK-NOT:   fir.convert %{{.*}} : (complex<f32>) -> f32
func.func @load_store_complex_as_real(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>, %v: f32) -> f32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, index) -> !fir.ref<complex<f32>>
  %cvt = fir.convert %elem : (!fir.ref<complex<f32>>) -> !fir.ref<f32>
  %decl = fir.declare %cvt {uniq_name = "rr"} : (!fir.ref<f32>) -> !fir.ref<f32>
  %load = fir.load %decl : !fir.ref<f32>
  fir.store %v to %decl : !fir.ref<f32>
  return %load : f32
}

// Reverse: real element passed as complex dummy. Peeling would load f32 then
// 'fir.convert' f32 -> complex, which is illegal.
// CHECK-LABEL: func.func @load_real_as_complex
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<f32>
// CHECK:       %[[CVT:.+]] = fir.convert %[[COOR]] : (!fir.ref<f32>) -> !fir.ref<complex<f32>>
// CHECK:       %[[M:.+]] = fir.convert %[[CVT]] : (!fir.ref<complex<f32>>) -> memref<complex<f32>>
// CHECK:       memref.load %[[M]][] : memref<complex<f32>>
// CHECK-NOT:   fir.convert %{{.*}} : (f32) -> complex<f32>
func.func @load_real_as_complex(%arg0: !fir.ref<!fir.array<4xf32>>) -> complex<f32> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %cvt = fir.convert %elem : (!fir.ref<f32>) -> !fir.ref<complex<f32>>
  %v = fir.load %cvt : !fir.ref<complex<f32>>
  return %v : complex<f32>
}

// Complex kind mismatch. fir.convert of complex values is legal, so peeling
// would silently fpext the whole value instead of a 16-byte load.
// CHECK-LABEL: func.func @load_complex4_as_complex8
// CHECK:       %[[M:.+]] = fir.convert %{{.+}} : (!fir.ref<complex<f64>>) -> memref<complex<f64>>
// CHECK:       memref.load %[[M]][] : memref<complex<f64>>
// CHECK-NOT:   fir.convert %{{.*}} : (complex<f32>) -> complex<f64>
func.func @load_complex4_as_complex8(%arg0: !fir.ref<!fir.array<4xcomplex<f32>>>) -> complex<f64> {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xcomplex<f32>>>, !fir.shape<1>, index) -> !fir.ref<complex<f32>>
  %cvt = fir.convert %elem : (!fir.ref<complex<f32>>) -> !fir.ref<complex<f64>>
  %v = fir.load %cvt : !fir.ref<complex<f64>>
  return %v : complex<f64>
}

//===----------------------------------------------------------------------===//
// Kind / TKR mismatch (silent numeric convert if peeled)
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @load_i32_as_f32
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<i32>
// CHECK:       %[[CVT:.+]] = fir.convert %[[COOR]] : (!fir.ref<i32>) -> !fir.ref<f32>
// CHECK:       %[[M:.+]] = fir.convert %[[CVT]] : (!fir.ref<f32>) -> memref<f32>
// CHECK:       memref.load %[[M]][] : memref<f32>
// CHECK-NOT:   fir.convert %{{.*}} : (i32) -> f32
func.func @load_i32_as_f32(%arg0: !fir.ref<!fir.array<4xi32>>) -> f32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %cvt = fir.convert %elem : (!fir.ref<i32>) -> !fir.ref<f32>
  %v = fir.load %cvt : !fir.ref<f32>
  return %v : f32
}

// CHECK-LABEL: func.func @load_f32_as_f64
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<f32>
// CHECK:       %[[CVT:.+]] = fir.convert %[[COOR]] : (!fir.ref<f32>) -> !fir.ref<f64>
// CHECK:       %[[M:.+]] = fir.convert %[[CVT]] : (!fir.ref<f64>) -> memref<f64>
// CHECK:       memref.load %[[M]][] : memref<f64>
// CHECK-NOT:   fir.convert %{{.*}} : (f32) -> f64
func.func @load_f32_as_f64(%arg0: !fir.ref<!fir.array<4xf32>>) -> f64 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  %cvt = fir.convert %elem : (!fir.ref<f32>) -> !fir.ref<f64>
  %v = fir.load %cvt : !fir.ref<f64>
  return %v : f64
}

// CHECK-LABEL: func.func @load_f64_as_f32
// CHECK:       %[[M:.+]] = fir.convert %{{.+}} : (!fir.ref<f32>) -> memref<f32>
// CHECK:       memref.load %[[M]][] : memref<f32>
// CHECK-NOT:   fir.convert %{{.*}} : (f64) -> f32
func.func @load_f64_as_f32(%arg0: !fir.ref<!fir.array<4xf64>>) -> f32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xf64>>, !fir.shape<1>, index) -> !fir.ref<f64>
  %cvt = fir.convert %elem : (!fir.ref<f64>) -> !fir.ref<f32>
  %v = fir.load %cvt : !fir.ref<f32>
  return %v : f32
}

// CHECK-LABEL: func.func @load_i32_as_i64
// CHECK:       %[[M:.+]] = fir.convert %{{.+}} : (!fir.ref<i64>) -> memref<i64>
// CHECK:       memref.load %[[M]][] : memref<i64>
// CHECK-NOT:   fir.convert %{{.*}} : (i32) -> i64
func.func @load_i32_as_i64(%arg0: !fir.ref<!fir.array<4xi32>>) -> i64 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %cvt = fir.convert %elem : (!fir.ref<i32>) -> !fir.ref<i64>
  %v = fir.load %cvt : !fir.ref<i64>
  return %v : i64
}

// Store of a narrower dummy. Peeling would memref.store i32 into memref<i64>.
// CHECK-LABEL: func.func @store_i32_as_i64
// CHECK:       %[[M:.+]] = fir.convert %{{.+}} : (!fir.ref<i64>) -> memref<i64>
// CHECK:       memref.store %arg1, %[[M]][] : memref<i64>
func.func @store_i32_as_i64(%arg0: !fir.ref<!fir.array<4xi32>>, %v: i64) {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %cvt = fir.convert %elem : (!fir.ref<i32>) -> !fir.ref<i64>
  fir.store %v to %cvt : !fir.ref<i64>
  return
}

//===----------------------------------------------------------------------===//
// logical / derived-type sequence association
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @load_logical_as_i32
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<!fir.logical<4>>
// CHECK:       %[[CVT:.+]] = fir.convert %[[COOR]] : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i32>
// CHECK:       %[[M:.+]] = fir.convert %[[CVT]] : (!fir.ref<i32>) -> memref<i32>
// CHECK:       memref.load %[[M]][] : memref<i32>
func.func @load_logical_as_i32(%arg0: !fir.ref<!fir.array<4x!fir.logical<4>>>) -> i32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4x!fir.logical<4>>>, !fir.shape<1>, index) -> !fir.ref<!fir.logical<4>>
  %cvt = fir.convert %elem : (!fir.ref<!fir.logical<4>>) -> !fir.ref<i32>
  %v = fir.load %cvt : !fir.ref<i32>
  return %v : i32
}

// Derived-type element passed as its first real component.
// CHECK-LABEL: func.func @load_derived_as_f32
// CHECK:       %[[COOR:.+]] = fir.array_coor %arg0{{.*}} : {{.*}} -> !fir.ref<!fir.type<_QTdt{x:f32,y:f32}>>
// CHECK:       %[[CVT:.+]] = fir.convert %[[COOR]] : (!fir.ref<!fir.type<_QTdt{x:f32,y:f32}>>) -> !fir.ref<f32>
// CHECK:       %[[M:.+]] = fir.convert %[[CVT]] : (!fir.ref<f32>) -> memref<f32>
// CHECK:       memref.load %[[M]][] : memref<f32>
func.func @load_derived_as_f32(%arg0: !fir.ref<!fir.array<4x!fir.type<_QTdt{x:f32,y:f32}>>>) -> f32 {
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %shape = fir.shape %c4 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape) %c1 : (!fir.ref<!fir.array<4x!fir.type<_QTdt{x:f32,y:f32}>>>, !fir.shape<1>, index) -> !fir.ref<!fir.type<_QTdt{x:f32,y:f32}>>
  %cvt = fir.convert %elem : (!fir.ref<!fir.type<_QTdt{x:f32,y:f32}>>) -> !fir.ref<f32>
  %v = fir.load %cvt : !fir.ref<f32>
  return %v : f32
}

//===----------------------------------------------------------------------===//
// Sequence association of a dummy array still peels the inner array_coor.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: func.func @seq_assoc_complex_elem_as_real_array
// CHECK:       %[[M:.+]] = fir.convert %{{.+}} : (!fir.ref<!fir.array<2xf32>>) -> memref<2xf32>
// CHECK:       memref.store %arg1, %[[M]][%{{.+}}] : memref<2xf32>
func.func @seq_assoc_complex_elem_as_real_array(%arg0: !fir.ref<!fir.array<1xcomplex<f32>>>, %v: f32) {
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %shape1 = fir.shape %c1 : (index) -> !fir.shape<1>
  %shape2 = fir.shape %c2 : (index) -> !fir.shape<1>
  %elem = fir.array_coor %arg0(%shape1) %c1 : (!fir.ref<!fir.array<1xcomplex<f32>>>, !fir.shape<1>, index) -> !fir.ref<complex<f32>>
  %cvt = fir.convert %elem : (!fir.ref<complex<f32>>) -> !fir.ref<!fir.array<2xf32>>
  %decl = fir.declare %cvt(%shape2) {uniq_name = "rr"} : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>) -> !fir.ref<!fir.array<2xf32>>
  %e1 = fir.array_coor %decl(%shape2) %c1 : (!fir.ref<!fir.array<2xf32>>, !fir.shape<1>, index) -> !fir.ref<f32>
  fir.store %v to %e1 : !fir.ref<f32>
  return
}
