// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  func.func @argument_only(%base: !xw.ptr<#xw.global>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "global", access = "read_only", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 8 : i32} {
    %root = xw.token : !xw.mem.token
    %value, %loaded = xw.load %base after %root : (!xw.ptr<#xw.global>, !xw.mem.token) -> (!xw.simd<i32, 8>, !xw.mem.token)
    return
  }

  func.func @local_argument(%base: !xw.ptr<#xw.local>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, address_space = "local", access = "read_write", size = 8, alignment = 8, offset = 24>
      ],
      xw.simd_width = 8 : i32} {
    %root = xw.token : !xw.mem.token
    %value, %loaded = xw.load %base after %root : (!xw.ptr<#xw.local>, !xw.mem.token) -> (!xw.simd<i32, 8>, !xw.mem.token)
    return
  }
}

// CHECK-NOT: llvm
// CHECK-LABEL: func.func @argument_only
// CHECK-SAME: xemachine.simd_size = 8 : i32
// CHECK-NOT: xemachine.uses_thread_ids
// CHECK: xemachine.mov {{.*}}execSize = 8{{.*}}src0Region = #xemachine.region<0, 1, 0>
// CHECK: xemachine.load_a64

// CHECK-LABEL: func.func @local_argument
// CHECK: xemachine.mov {{.*}}src0Sub = 6 : i32{{.*}}i32
// CHECK: xemachine.load_slm
