// RUN: inter-opt --split-input-file --inter-convert-calls -verify-diagnostics %s

module {
  func.func @direct_call() attributes {xemachine.kernel} {
    // expected-error@+1 {{function calls are not supported; 'ordinary_function' is not a recognized builtin}}
    llvm.call @ordinary_function() : () -> ()
    return
  }
  llvm.func @ordinary_function()
}

// -----

module {
  func.func @indirect_call(%callee: !llvm.ptr) attributes {xemachine.kernel} {
    // expected-error@+1 {{indirect function calls are not supported}}
    llvm.call %callee() : !llvm.ptr, () -> ()
    return
  }
}

// -----

module {
  func.func @builtin_prefix() attributes {xemachine.kernel} {
    %dimension = llvm.mlir.constant(0 : i32) : i32
    // expected-error@+1 {{function calls are not supported; '_Z13get_global_idj.not_a_builtin' is not a recognized builtin}}
    %id = llvm.call spir_funccc @_Z13get_global_idj.not_a_builtin(%dimension)
        : (i32) -> i64
    return
  }
  llvm.func spir_funccc @_Z13get_global_idj.not_a_builtin(i32) -> i64
}
