/// Verify that converts are only generated one per fir.declare

// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// Derived from 
//  do i = 1, n
//    a(i) = i
//    call y(i)
//    call z(i)
//  enddo

// CHECK-LABEL: func.func @x_
// CHECK:       [[ALLOCA:%.+]] = memref.alloca() {bindc_name = "i"} : memref<i32>
// CHECK-NEXT:  [[CONVERT0:%[0-9]]] = fir.convert [[ALLOCA]] : (memref<i32>) -> !fir.ref<i32>
// CHECK-NEXT:  [[DECLARE:%[0-9]]] = fir.declare [[CONVERT0]]
// CHECK-NEXT:  [[CONVERT1:%[0-9]]] = fir.convert [[DECLARE]] : (!fir.ref<i32>) -> memref<i32>
// CHECK-NEXT:  [[LOAD:%[0-9]]] = memref.load [[CONVERT1]]
// CHECK-NEXT:  [[CONVERT2:%[0-9]]] = fir.convert [[CONVERT1]] : (memref<i32>) -> !fir.ref<i32>
// CHECK-NEXT:  fir.call @y_([[CONVERT2]])
// CHECK-NEXT:  [[CONVERT3:%[0-9]]] = fir.convert [[CONVERT1]] : (memref<i32>) -> !fir.ref<i32>
// CHECK-NEXT:  fir.call @z_([[CONVERT3]])
func.func @x_() attributes {fir.internal_name = "_QPx"} {
  %1 = fir.alloca i32 {bindc_name = "i"}
  %2 = fir.declare %1 {uniq_name = "_QFxEi"} : (!fir.ref<i32>) -> !fir.ref<i32>
  %3 = fir.load %2 : !fir.ref<i32>
  fir.call @y_(%2) fastmath<contract> : (!fir.ref<i32>) -> ()
  fir.call @z_(%2) fastmath<contract> : (!fir.ref<i32>) -> ()
  return
}
