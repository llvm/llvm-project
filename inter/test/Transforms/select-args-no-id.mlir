// RUN: inter-opt %s --inter-select-to-machine | FileCheck %s

module {
  // CHECK-LABEL: func.func @argument_only
  // CHECK-SAME: xemachine.inline_data_payload_size = 32 : i32
  // CHECK-SAME: xemachine.kernel_args = [#xemachine.kernel_arg<kind = by_pointer, offset = 24, size = 8>]
  // CHECK-NOT: xemachine.uses_thread_ids
  func.func @argument_only(%base: !llvm.ptr<1>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, offset = 24, size = 8>
      ]} {
    %root = xw.token
    %value, %loaded = xw.load %base dep %root : !llvm.ptr<1> -> i32
    return
  }

  // CHECK: xemachine.load_block_a32
  // CHECK: xemachine.load_a64
}
