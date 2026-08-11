// RUN: inter-opt --split-input-file --inter-select-to-machine -verify-diagnostics %s

module {
  // expected-error@+1 {{kernel argument descriptor does not match type}}
  func.func @wrong_kind(%arg: !llvm.ptr<1>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_value, offset = 24, size = 8>
      ]} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument overlaps the implicit payload}}
  func.func @reserved(%arg: !llvm.ptr<1>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, offset = 16, size = 8>
      ]} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument payload is misaligned}}
  func.func @misaligned(%arg: !llvm.ptr<1>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, offset = 28, size = 8>
      ]} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument is outside the loaded payload}}
  func.func @out_of_bounds(%arg: !llvm.ptr<1>) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, offset = 64, size = 8>
      ]} {
    return
  }
}

// -----

module {
  // expected-error@+1 {{kernel argument payloads overlap}}
  func.func @overlap(%pointer: !llvm.ptr<1>, %value: i32) attributes {
      xemachine.kernel,
      xemachine.kernel_args = [
        #xemachine.kernel_arg<kind = by_pointer, offset = 24, size = 8>,
        #xemachine.kernel_arg<kind = by_value, offset = 28, size = 4>
      ]} {
    return
  }
}
