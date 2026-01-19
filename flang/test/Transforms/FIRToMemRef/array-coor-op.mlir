// Test conversions to index
// Test dynamically legal array coor ops, we should both keep a fir.array_coor
// and generate a memref.store
// example fortran code:
//  subroutine func(a)
//      integer :: a(4)
//      end
//  
//  subroutine slice(a)
//    integer :: a(8)
//    call func(a(1:4))
//    a(5) = 1
//    a(1) = 2
//  end
// RUN: fir-opt %s --fir-to-memref --allow-unregistered-dialect | FileCheck %s

// CHECK-LABEL: func.func @slice
// CHECK:       [[C5:%.+]]       = arith.constant 5 : index
// CHECK:       [[C1_I32:%.+]]   = arith.constant 1 : i32
// CHECK:       [[C1:%.+]]       = arith.constant 1 : index
// CHECK:       [[C8:%.+]]       = arith.constant 8 : index
// CHECK:       [[DUMMY:%.+]]    = fir.dummy_scope : !fir.dscope
// CHECK:       [[SHAPE:%.+]]    = fir.shape [[C8]] : (index) -> !fir.shape<1>
// CHECK:       [[DECLARE:%.+]]  = fir.declare %arg0([[SHAPE]]) dummy_scope [[DUMMY]]
// CHECK:       [[COOR1:%.+]]    = fir.array_coor [[DECLARE]]([[SHAPE]]) [[C1]] : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
// CHECK:       [[SLICE:%.+]]    = fir.convert [[COOR1]] : (!fir.ref<i32>) -> !fir.ref<!fir.array<4xi32>>
// CHECK:       [[CONVERT:%.+]]  = fir.convert [[DECLARE]] : (!fir.ref<!fir.array<8xi32>>) -> memref<8xi32>
// CHECK:       [[C1B:%.+]]      = arith.constant 1 : index
// CHECK:       [[SUB:%.+]]      = arith.subi [[C5]], [[C1B]] : index
// CHECK:       [[MUL:%.+]]      = arith.muli [[SUB]], [[C1B]] : index
// CHECK:       [[SUB2:%.+]]     = arith.subi [[C1B]], [[C1B]] : index
// CHECK:       [[IDX:%.+]]      = arith.addi [[MUL]], [[SUB2]] : index
// CHECK:       memref.store [[C1_I32]], [[CONVERT]][[[IDX]]] : memref<8xi32>
func.func @slice(%arg0: !fir.ref<!fir.array<8xi32>>){
  %c5 = arith.constant 5 : index
  %c1_i32 = arith.constant 1 : i32
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %0 = fir.dummy_scope : !fir.dscope
  %1 = fir.shape %c8 : (index) -> !fir.shape<1>
  %2 = fir.declare %arg0(%1) dummy_scope %0 {uniq_name = "_QFsliceEa"} : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, !fir.dscope) -> !fir.ref<!fir.array<8xi32>>
  %3 = fir.array_coor %2(%1) %c1 : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  %4 = fir.convert %3 : (!fir.ref<i32>) -> !fir.ref<!fir.array<4xi32>>
  %5 = fir.array_coor %2(%1) %c5 : (!fir.ref<!fir.array<8xi32>>, !fir.shape<1>, index) -> !fir.ref<i32>
  fir.store %c1_i32 to %5 : !fir.ref<i32>
  return                                                                                                                                                                                                                               
}
