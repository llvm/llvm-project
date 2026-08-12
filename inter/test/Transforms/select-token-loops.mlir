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
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @token_loops
// CHECK: xemachine.uniform_loop
// CHECK: xemachine.continue_if
// CHECK: xemachine.uniform_loop
// CHECK: xemachine.uniform_if
// CHECK: xemachine.eot
