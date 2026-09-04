// Test conversions of fir.alloca
// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// Test that compiler generated fir.allocas are converted, slightly edited from
//   subroutine alloca(a)
//     call f(1)
//   end

// CHECK-LABEL: func.func @alloca
// CHECK:       [[ALLOCA:%.+]] = memref.alloca() : memref<i32>
// CHECK-NEXT:  [[CONVERT0:%[0-9]]] = fir.convert [[ALLOCA]] : (memref<i32>) -> !fir.ref<i32>
// CHECK:       [[CONSTANT:%.+]] = arith.constant 1 : i32
// CHECK:       memref.store [[CONSTANT]], [[ALLOCA]][] : memref<i32>
// CHECK:       fir.call @f([[CONVERT0]])

func.func @alloca(%arg0: !fir.ref<f32> {fir.bindc_name = "a"}) {
  %0 = fir.alloca i32 {adapt.valuebyref}
  %c1_i32 = arith.constant 1 : i32
  %1 = fir.dummy_scope : !fir.dscope
  %2 = fir.declare %arg0 dummy_scope %1 {uniq_name = "alloca"} : (!fir.ref<f32>, !fir.dscope) -> !fir.ref<f32>
  fir.store %c1_i32 to %0 : !fir.ref<i32>
  %false = arith.constant false
  fir.call @f(%0) fastmath<contract> : (!fir.ref<i32>) -> ()
  return
}
func.func private @f(!fir.ref<i32>)

// Test that compiler generated fir.allocas are converted, slightly edited from
// subroutine passbyvalue(x)
//   integer, value :: x
// end subroutine

// CHECK-LABEL: func.func @passbyvalue
// CHECK:       [[ALLOCA:%.+]] = memref.alloca() : memref<i32>
// CHECK-NEXT:  [[CONVERT0:%[0-9]]] = fir.convert [[ALLOCA]] : (memref<i32>) -> !fir.ref<i32>
// CHECK-NEXT:  memref.store %arg0, [[ALLOCA]][] : memref<i32>

func.func @passbyvalue(%arg0: i32 {fir.bindc_name = "x"}) {
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.alloca i32
  fir.store %arg0 to %1 : !fir.ref<i32>
  %2 = fir.declare %1 dummy_scope %0 {fortran_attrs = #fir.var_attrs<value>, uniq_name = "_QFpEx"} : (!fir.ref<i32>, !fir.dscope) -> !fir.ref<i32>
  return
}

// Multi-dim dynamic alloca, reduced from
// subroutine alloca_2d(n,m)
//   real :: a(n,m)
// end

// CHECK-LABEL: func.func @_QPalloca_2d
// CHECK:       [[CON5:%.+]] = arith.constant 5
// CHECK-NEXT:  [[CON3:%.+]] = arith.constant 3
// CHECK-NEXT:  [[ALLOCA:%.+]] = memref.alloca([[CON3]], [[CON5]]) {bindc_name = "a"} : memref<?x?xf32>
// CHECK-NEXT:  [[CONVERT0:%[0-9]]] = fir.convert [[ALLOCA]] : (memref<?x?xf32>) -> !fir.ref<!fir.array<?x?xf32>>

func.func @_QPalloca_2d() {
  %0 = arith.constant 5 : index
  %1 = arith.constant 3 : index
  %2 = fir.alloca !fir.array<?x?xf32>, %0, %1 {bindc_name = "a"}
  return
}

// CHECK-LABEL: func.func @alloca_nonconvertible
// CHECK-NEXT:  [[ALLOCA1:%.+]] = fir.alloca
// CHECK-NEXT:  [[ALLOCA2:%.+]] = memref.alloca
// CHECK-NEXT:  [[CONVERT0:%[0-9]]] = fir.convert [[ALLOCA2]] : (memref<i32>) -> !fir.ref<i32>
func.func @alloca_nonconvertible() {
  %0 = fir.alloca !fir.box<!fir.array<20xi32>>
  %1 = fir.alloca i32
  return
}

// compiler generated alloca will not preserve declare
// CHECK-LABEL: func.func @peep_declare
// CHECK:       [[ALLOCA:%.+]] = memref.alloca() : memref<i32>
// CHECK-NOT:   fir.declare
// CHECK:       memref.store %c1_i32, [[ALLOCA]][] : memref<i32>
func.func @peep_declare() {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32
  %1 = fir.declare %0 {uniq_name = "some_name"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c1_i32 to %1 : !fir.ref<i32>
  return
}

// check that attributes are copied when they exist
// CHECK-LABEL: func.func @preserve_declare
// CHECK:       [[ALLOCA:%.*]] = memref.alloca() {bindc_name = "x", uniq_name = "y"} : memref<i32>
// CHECK-NOT:   memref.store %c1_i32, [[ALLOCA]][] : memref<i32>
func.func @preserve_declare() {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {bindc_name = "x", uniq_name = "y"}
  %1 = fir.declare %0 {uniq_name = "some_name"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c1_i32 to %1 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @copy_cuf_data_attr
// CHECK:       memref.alloca() {cuf.data_attr = #cuf.cuda<device>} : memref<i32>
func.func @copy_cuf_data_attr() {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {cuf.data_attr = #cuf.cuda<device>}
  %1 = fir.declare %0 {uniq_name = "some_name"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c1_i32 to %1 : !fir.ref<i32>
  return
}

// CHECK-LABEL: func.func @copy_acc_var_name_attr
// CHECK:       memref.alloca() {acc.var_name = #acc.var_name<"x">} : memref<i32>
func.func @copy_acc_var_name_attr() {
  %c1_i32 = arith.constant 1 : i32
  %0 = fir.alloca i32 {acc.var_name = #acc.var_name<"x">}
  %1 = fir.declare %0 {uniq_name = "some_name"} : (!fir.ref<i32>) -> !fir.ref<i32>
  fir.store %c1_i32 to %1 : !fir.ref<i32>
  return
}
