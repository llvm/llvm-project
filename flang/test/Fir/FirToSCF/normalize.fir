// Test normalization behavior of fir.do_loop lowering to scf.
//
// RUN: fir-opt %s --fir-to-scf --allow-unregistered-dialect | FileCheck %s

// Test that loops with unknown step are normalized.
//
// e.g.
// subroutine loop(a,lb,ub,step)
//   integer :: a(8), lb, ub, step
//   do i = lb, ub, step
//     a(i) = i
//   enddo
// end

// CHECK-LABEL: func.func @unknown_step_
// CHECK:       [[LB:%[0-9]+]] = arith.index_cast %arg0
// CHECK:       [[UB:%[0-9]+]] = arith.index_cast %arg1
// CHECK:       [[STEP:%[0-9]+]] = arith.index_cast %arg2
// CHECK:       [[SUB:%[0-9]+]] = arith.subi [[UB]], [[LB]] : index
// CHECK-NEXT:  [[ADD:%[0-9]+]] = arith.addi [[SUB]], [[STEP]] : index
// CHECK-NEXT:  [[DIV:%[0-9]+]] = arith.divsi [[ADD]], [[STEP]] : index
// CHECK:       scf.for %arg4 = %c0{{.*}} to [[DIV]] step %c1{{.*}}
// CHECK-NEXT:    [[IVMUL:%[0-9]+]] = arith.muli %arg4, [[STEP]] : index
// CHECK-NEXT:    [[IVADD:%[0-9]+]] = arith.addi [[LB]], [[IVMUL]] : index
// CHECK-NEXT:    memref.store [[IVADD]], %arg3[]

func.func @unknown_step_(%arg0: i64, %arg1: i64, %arg2: i64, %arg3: memref<index>) {
  %0 = arith.index_cast %arg0 : i64 to index
  %1 = arith.index_cast %arg1 : i64 to index
  %2 = arith.index_cast %arg2 : i64 to index
  fir.do_loop %arg4 = %0 to %1 step %2 {
    memref.store %arg4, %arg3[] : memref<index>
  }
  return
}

// Test that loops with known negative step are normalized.
//
// e.g.
// subroutine loop(a)
//   integer :: a(8)
//   do i = 8, 1, -1
//     a(i) = i
//   enddo
// end

// CHECK-LABEL: func.func @negative_step_
// CHECK:       [[LB:%.+]] = arith.constant 8
// CHECK:       [[UB:%.+]] = arith.constant 1
// CHECK:       [[STEP:%.+]] = arith.constant -1
// CHECK:       [[SUB:%[0-9]+]] = arith.subi [[UB]], [[LB]] : index
// CHECK-NEXT:  [[ADD:%[0-9]+]] = arith.addi [[SUB]], [[STEP]] : index
// CHECK-NEXT:  [[DIV:%[0-9]+]] = arith.divsi [[ADD]], [[STEP]] : index
// CHECK:       scf.for %arg1 = %c0{{.*}} to [[DIV]] step %c1{{.*}}
// CHECK-NEXT:    [[IVMUL:%[0-9]+]] = arith.muli %arg1, [[STEP]] : index
// CHECK-NEXT:    [[IVADD:%[0-9]+]] = arith.addi [[LB]], [[IVMUL]] : index
// CHECK-NEXT:    memref.store [[IVADD]], %arg0[]

func.func @negative_step_(%arg0: memref<index>) {
  %c8 = arith.constant 8 : index
  %c1 = arith.constant 1 : index
  %c-1 = arith.constant -1 : index
  fir.do_loop %arg1 = %c8 to %c1 step %c-1 {
    memref.store %arg1, %arg0[] : memref<index>
  }
  return
}


