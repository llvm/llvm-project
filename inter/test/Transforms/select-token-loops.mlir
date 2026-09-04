// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @token_loops() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %zero = xw.constant 0 : index
    %one = xw.constant 1 : index
    %two = xw.constant 2 : index
    %allocation = xw.alloc() {bytesize = 16 : i64, align = 16 : i64}
        : !xw.ptr<#xw.local>
    %root = xw.token : !xw.mem.token
    %for_token = scf.for %iv = %zero to %two step %one
        iter_args(%token = %root) -> (!xw.mem.token) {
      %next = xw.alloc_release %allocation after %token
          : (!xw.ptr<#xw.local>, !xw.mem.token) -> !xw.mem.token
      scf.yield %next : !xw.mem.token
    }
    %while_token = scf.while (%token = %for_token) : (!xw.mem.token) -> !xw.mem.token {
      %condition = xw.cmpi ne %zero, %one : index, index -> i1
      scf.condition(%condition) %token : !xw.mem.token
    } do {
    ^bb0(%token: !xw.mem.token):
      %next = xw.alloc_release %allocation after %token
          : (!xw.ptr<#xw.local>, !xw.mem.token) -> !xw.mem.token
      scf.yield %next : !xw.mem.token
    }
    return
  }

  func.func @asymmetric_while() attributes {
      xemachine.kernel, xemachine.kernel_args = [],
      xw.simd_width = 8 : i32} {
    %zero = xw.constant 0 : index
    %one = xw.constant 1 : index
    %two = xw.constant 2 : index
    %initial = xw.constant 0 : i32 -> !xw.simd<i32, 8>
    %result:3 = scf.while (%iv = %zero, %iter = %initial)
        : (index, !xw.simd<i32, 8>)
        -> (index, !xw.simd<i32, 8>, !xw.simd<i32, 8>) {
      %condition = xw.cmpi slt %iv, %two : index, index -> i1
      %next = xw.binary addi %iv, %one : index, index -> index
      scf.condition(%condition) %next, %iter, %iter
          : index, !xw.simd<i32, 8>, !xw.simd<i32, 8>
    } do {
    ^bb0(%iv: index, %iter: !xw.simd<i32, 8>, %exit: !xw.simd<i32, 8>):
      scf.yield %iv, %iter : index, !xw.simd<i32, 8>
    }
    return
  }
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @token_loops
// CHECK: xemachine.uniform_loop
// CHECK: xemachine.continue_if
// CHECK: xemachine.uniform_loop
// CHECK: [[COND:%.*]] = xemachine.cmp
// CHECK-NEXT: [[SNAPSHOT:%.*]] = xemachine.mov [[COND]]
// CHECK: [[BODY_COND:%.*]] = xemachine.cmp ne [[SNAPSHOT]]
// CHECK-NEXT: xemachine.uniform_if [[BODY_COND]]
// CHECK: [[CONTINUE:%.*]] = xemachine.cmp ne [[SNAPSHOT]]
// CHECK-NEXT: xemachine.continue_if [[CONTINUE]]
// CHECK: xemachine.eot

// CHECK-LABEL: func.func @asymmetric_while
// CHECK: xemachine.uniform_loop
// CHECK: ^bb0([[IV:%.*]]: {{.*}}, [[ITER:%.*]]: !xemachine.reg<8, -1>, [[EXIT:%.*]]: !xemachine.reg<8, -1>):
// CHECK-NEXT: [[EXIT_SNAPSHOT:%.*]] = xemachine.mov [[ITER]]
// CHECK-SAME: execSize = 8 : i32
// CHECK: [[BODY:%.*]]:2 = xemachine.uniform_if
// CHECK: xemachine.continue_if {{.*}}([[BODY]]#0, [[BODY]]#1, [[EXIT_SNAPSHOT]]
