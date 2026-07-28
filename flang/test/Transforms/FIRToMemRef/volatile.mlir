// RUN: fir-opt %s --fir-to-memref | FileCheck %s

// memref dialect currently has no way to express volatile loads and stores.

// CHECK-LABEL: func.func @volatile_scalar_dummy
// CHECK:         %[[CAST:.*]] = fir.volatile_cast %arg0 : (!fir.ref<f128>) -> !fir.ref<f128, volatile>
// CHECK:         %[[DECL:.*]] = fir.declare %[[CAST]]
// CHECK:         %[[CONV1:.*]] = fir.convert %[[DECL]] :  (!fir.ref<f128, volatile>) -> memref<f128>
// CHECK:         %[[LOAD:.*]] = memref.load %[[CONV1]][] : memref<f128>
// CHECK:         %[[CONV2:.*]] = fir.convert %[[DECL]] :  (!fir.ref<f128, volatile>) -> memref<f128>
// CHECK:         memref.store %[[LOAD]], %[[CONV2]][] : memref<f128>
func.func @volatile_scalar_dummy(%arg0: !fir.ref<f128>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.volatile_cast %arg0 : (!fir.ref<f128>) -> !fir.ref<f128, volatile>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "x"} : (!fir.ref<f128, volatile>, !fir.dscope) -> !fir.ref<f128, volatile>
  %3 = fir.load %2 : !fir.ref<f128, volatile>
  fir.store %3 to %2 : !fir.ref<f128, volatile>
  return
}

// CHECK-LABEL: func.func @volatile_local
// CHECK:         %[[ALLOCA:.*]] = memref.alloca() {bindc_name = "i", uniq_name = "i"} : memref<i32>
// CHECK:         %[[CONV:.*]] = fir.convert %[[ALLOCA]] : (memref<i32>) -> !fir.ref<i32>
// CHECK:         %[[VCAST:.*]] = fir.volatile_cast %[[CONV]] : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
// CHECK:         %[[DECL:.*]] = fir.declare %[[VCAST]] {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "i"}
// CHECK:         %[[CONV1:.*]] = fir.convert %[[DECL]] : (!fir.ref<i32, volatile>) -> memref<i32>
// CHECK:         memref.store %{{.*}}, %[[CONV1]][] : memref<i32>
// CHECK:         %[[CONV2:.*]] = fir.convert %[[DECL]] : (!fir.ref<i32, volatile>) -> memref<i32>
// CHECK:         %[[LOAD:.*]] = memref.load %[[CONV2]][] : memref<i32>
func.func @volatile_local() {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "i"}
  %1 = fir.volatile_cast %0 : (!fir.ref<i32>) -> !fir.ref<i32, volatile>
  %2 = fir.declare %1 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "i"} : (!fir.ref<i32, volatile>) -> !fir.ref<i32, volatile>
  fir.store %c1_i32 to %2 : !fir.ref<i32, volatile>
  %3 = fir.load %2 : !fir.ref<i32, volatile>
  return
}

// CHECK-LABEL: func.func @volatile_array_element
// CHECK:         %[[VCAST:.*]] = fir.volatile_cast %arg0 : (!fir.ref<!fir.array<3xf32>>) -> !fir.ref<!fir.array<3xf32>, volatile>
// CHECK:         %[[DECL:.*]] = fir.declare %[[VCAST]](%{{.*}}) dummy_scope {{.*}} {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "a"} : (!fir.ref<!fir.array<3xf32>, volatile>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xf32>, volatile>
// CHECK:         %[[CONV:.*]] = fir.convert %[[DECL]] :  (!fir.ref<!fir.array<3xf32>, volatile>) -> memref<3xf32>
// CHECK:         %[[LOAD:.*]] = memref.load %[[CONV]][%{{.*}}] : memref<3xf32>
func.func @volatile_array_element(%arg0: !fir.ref<!fir.array<3xf32>>) {
  %c1 = arith.constant 1 : index
  %c3 = arith.constant 3 : index
  %0 = fir.undefined !fir.dscope
  %shape = fir.shape %c3 : (index) -> !fir.shape<1>
  %1 = fir.volatile_cast %arg0 : (!fir.ref<!fir.array<3xf32>>) -> !fir.ref<!fir.array<3xf32>, volatile>
  %2 = fir.declare %1(%shape) dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "a"} : (!fir.ref<!fir.array<3xf32>, volatile>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<3xf32>, volatile>
  %3 = fir.array_coor %2(%shape) %c1 : (!fir.ref<!fir.array<3xf32>, volatile>, !fir.shape<1>, index) -> !fir.ref<f32, volatile>
  %4 = fir.load %3 : !fir.ref<f32, volatile>
  return
}

// CHECK-LABEL: func.func @mixed_volatile_and_plain
// CHECK:         %[[VCAST:.*]] = fir.volatile_cast %arg0 : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
// CHECK:         %[[VDECL:.*]] = fir.declare %[[VCAST]] dummy_scope %{{.*}} {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "v"} : (!fir.ref<f32, volatile>, !fir.dscope) -> !fir.ref<f32, volatile>
// CHECK:         %[[PDECL:.*]] = fir.declare %arg1 dummy_scope %{{.*}} {uniq_name = "p"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
// CHECK:         %[[VCONV:.*]] = fir.convert %[[VDECL]] : (!fir.ref<f32, volatile>) -> memref<f32>
// CHECK:         %[[VLOAD:.*]] = memref.load %[[VCONV]][] : memref<f32>
// CHECK:         %[[PCONV:.*]] = fir.convert %[[PDECL]] : (!fir.ref<f32>) -> memref<f32>
// CHECK:         memref.store %[[VLOAD]], %[[PCONV]][] : memref<f32>
func.func @mixed_volatile_and_plain(%arg0: !fir.ref<f32>, %arg1: !fir.ref<f32>) {
  %0 = fir.undefined !fir.dscope
  %1 = fir.volatile_cast %arg0 : (!fir.ref<f32>) -> !fir.ref<f32, volatile>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<volatile>, uniq_name = "v"} : (!fir.ref<f32, volatile>, !fir.dscope) -> !fir.ref<f32, volatile>
  %3 = fir.declare %arg1 dummy_scope %0 {uniq_name = "p"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  %4 = fir.load %2 : !fir.ref<f32, volatile>
  fir.store %4 to %3 : !fir.ref<f32>
  return
}
