// RUN: inter-opt --split-input-file --inter-select-to-machine -verify-diagnostics %s

module {
  llvm.mlir.global internal @global(0 : i32) : i32

  func.func @unsupported_address_space() attributes {
      xemachine.kernel, xemachine.kernel_args = []} {
    // expected-error@+1 {{unsupported pointer address space}}
    %address = llvm.mlir.addressof @global : !llvm.ptr
    %root = xw.token
    %value, %loaded = xw.load %address dep %root : !llvm.ptr -> i32
    return
  }
}

// -----

module {
  func.func @wide_store(%address: !llvm.ptr<1>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, offset = 24, size = 8>
      ]} {
    %value = llvm.mlir.constant(0 : i64) : i64
    %root = xw.token
    // expected-error@+1 {{only i32 stores are selected}}
    %stored = xw.store %address, %value dep %root
        : !llvm.ptr<1>, i64 -> !xemachine.mem.token
    return
  }
}

// -----

module {
  func.func @slm_atomic() attributes {
      xemachine.kernel, xemachine.kernel_args = []} {
    %address = llvm.mlir.addressof @slm : !llvm.ptr<3>
    %one = llvm.mlir.constant(1 : i32) : i32
    %root = xw.token
    // expected-error@+1 {{atomic address must be a global pointer}}
    %old, %atomic = xw.atomic_add %address, %one dep %root
        : !llvm.ptr<3>, i32 -> (i32, !xemachine.mem.token)
    return
  }
  llvm.mlir.global internal @slm(0 : i32) {addr_space = 3 : i32} : i32
}
