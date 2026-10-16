// RUN: inter-opt --split-input-file --inter-import-llvm \
// RUN:   --inter-convert-llvm-to-xw -verify-diagnostics %s

module {
  llvm.func spir_kernelcc @local_to_generic(%local: !llvm.ptr<3>) {
    // expected-error@+2 {{local and generic address-space casts require provenance-preserving selection}}
    // expected-error@+1 {{failed to legalize operation 'llvm.addrspacecast'}}
    %generic = llvm.addrspacecast %local : !llvm.ptr<3> to !llvm.ptr<4>
    llvm.return
  }
}

// -----

module {
  llvm.func spir_kernelcc @atomic_order(%pointer: !llvm.ptr<1>, %value: i32) {
    // expected-error@+2 {{only monotonic LLVM atomic RMW ordering is supported}}
    // expected-error@+1 {{failed to legalize operation 'llvm.atomicrmw'}}
    %old = llvm.atomicrmw add %pointer, %value acquire : !llvm.ptr<1>, i32
    llvm.return
  }
}

// -----

module {
  llvm.func spir_kernelcc @atomic_scope(%pointer: !llvm.ptr<1>, %value: i32) {
    // expected-error@+2 {{LLVM atomic RMW syncscope has no exact XW representation}}
    // expected-error@+1 {{failed to legalize operation 'llvm.atomicrmw'}}
    %old = llvm.atomicrmw add %pointer, %value syncscope("workgroup") monotonic
        : !llvm.ptr<1>, i32
    llvm.return
  }
}

// -----

module {
  llvm.func spir_kernelcc @fence() {
    // expected-error@+2 {{LLVM fence ordering and scope have no exact XW representation}}
    // expected-error@+1 {{failed to legalize operation 'llvm.fence'}}
    llvm.fence seq_cst
    llvm.return
  }
}

// -----

module {
  llvm.func spir_kernelcc @atomic_kind(%pointer: !llvm.ptr<1>, %value: i32) {
    // expected-error@+2 {{only integer add LLVM atomic RMW is supported}}
    // expected-error@+1 {{failed to legalize operation 'llvm.atomicrmw'}}
    %old = llvm.atomicrmw xchg %pointer, %value monotonic : !llvm.ptr<1>, i32
    llvm.return
  }
}

// -----

module {
  llvm.func spir_kernelcc @volatile_load(%pointer: !llvm.ptr<1>) {
    // expected-error@+2 {{volatile LLVM load has no exact XW representation}}
    // expected-error@+1 {{failed to legalize operation 'llvm.load'}}
    %value = llvm.load volatile %pointer : !llvm.ptr<1> -> i32
    llvm.return
  }
}

// -----

module {
  llvm.func spir_kernelcc @atomic_store(%pointer: !llvm.ptr<1>, %value: i32) {
    // expected-error@+2 {{atomic LLVM store has no exact XW representation}}
    // expected-error@+1 {{failed to legalize operation 'llvm.store'}}
    llvm.store %value, %pointer atomic monotonic {alignment = 4 : i64}
        : i32, !llvm.ptr<1>
    llvm.return
  }
}
