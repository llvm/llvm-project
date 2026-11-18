// RUN: fir-opt -split-input-file --mlir-disable-threading \
// RUN:   -pass-pipeline='builtin.module(test-fir-alias-analysis-modref)' \
// RUN:   %s -o /dev/null 2>&1 | FileCheck %s

// Test fir.call modref for a saved local variable of a procedure. All
// instances of the procedure share the variable, so a call that may invoke
// the procedure may access the variable. The call cannot invoke the procedure
// if the callee is PURE, or if the call is in the function of the procedure
// itself and the procedure is NON_RECURSIVE. The main program cannot be
// referenced at all.

// A RECURSIVE procedure: bump may invoke work, which may modify x. The PURE
// pf cannot invoke the impure work.
// CHECK-LABEL: Testing : "recursive_owner"
// CHECK: bump_call -> x#0: ModRef{{$}}
// CHECK: pf_call -> x#0: NoModRef{{$}}
module @recursive_owner {
  fir.global internal @_QMmFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QMmPwork() attributes {fir.proc_attrs = #fir.proc_attrs<recursive>} {
    %0 = fir.address_of(@_QMmFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QMmPbump() {test.ptr = "bump_call"} : () -> ()
    fir.call @_QMmPpf() proc_attrs<pure> {test.ptr = "pf_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QMmPbump()
  func.func private @_QMmPpf() attributes {fir.proc_attrs = #fir.proc_attrs<pure>}
}

// -----

// A procedure that is neither RECURSIVE nor NON_RECURSIVE may be recursive.
// CHECK-LABEL: Testing : "default_owner"
// CHECK: bump_call -> x#0: ModRef{{$}}
// CHECK: pf_call -> x#0: NoModRef{{$}}
module @default_owner {
  fir.global internal @_QMmFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QMmPwork() {
    %0 = fir.address_of(@_QMmFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QMmPbump() {test.ptr = "bump_call"} : () -> ()
    fir.call @_QMmPpf() proc_attrs<pure> {test.ptr = "pf_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QMmPbump()
  func.func private @_QMmPpf() attributes {fir.proc_attrs = #fir.proc_attrs<pure>}
}

// -----

// A NON_RECURSIVE procedure: bump cannot invoke work.
// CHECK-LABEL: Testing : "non_recursive_owner"
// CHECK: bump_call -> x#0: NoModRef{{$}}
// CHECK: pf_call -> x#0: NoModRef{{$}}
module @non_recursive_owner {
  fir.global internal @_QMmFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QMmPwork() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %0 = fir.address_of(@_QMmFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QMmPbump() {test.ptr = "bump_call"} : () -> ()
    fir.call @_QMmPpf() proc_attrs<pure> {test.ptr = "pf_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QMmPbump()
  func.func private @_QMmPpf() attributes {fir.proc_attrs = #fir.proc_attrs<pure>}
}

// -----

// The NON_RECURSIVE work is inlined twice into caller: no instance of work is
// active at the call between the two inlined bodies, so bump may invoke work,
// which may modify x.
//
//   subroutine caller()
//     use m
//     call work()   ! x = 1
//     call bump()
//     call work()   ! reads x
//   end subroutine
// CHECK-LABEL: Testing : "inlined_owner"
// CHECK: bump_call -> x1#0: ModRef{{$}}
// CHECK: bump_call -> x2#0: ModRef{{$}}
module @inlined_owner {
  fir.global internal @_QMmFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QPcaller() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %c1 = arith.constant 1 : i32
    %0 = fir.address_of(@_QMmFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x1"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.store %c1 to %1 : !fir.ref<i32>
    fir.call @_QMmPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x2"} : (!fir.ref<i32>) -> !fir.ref<i32>
    %3 = fir.load %2 : !fir.ref<i32>
    return
  }
  func.func private @_QMmPwork() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>}
  func.func private @_QMmPbump()
}

// -----

// The NON_RECURSIVE m::work is inlined into n::work, a procedure with the same
// name in another module.
// CHECK-LABEL: Testing : "other_module"
// CHECK: bump_call -> x#0: ModRef{{$}}
module @other_module {
  fir.global internal @_QMmFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QMnPwork() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %0 = fir.address_of(@_QMmFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QMmPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QMmPbump()
}

// -----

// A procedure named after the BIND(C) binding label of work_c does not own
// the saved local of work_c.
// CHECK-LABEL: Testing : "bind_c_owner"
// CHECK: bump_call -> z#0: ModRef{{$}}
module @bind_c_owner {
  fir.global internal @_QFwork_cEz : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @workC() attributes {fir.bindc_name = "workC", fir.proc_attrs = #fir.proc_attrs<non_recursive, bind_c>} {
    %0 = fir.address_of(@_QFwork_cEz) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFwork_cEz") {test.ptr = "z"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPbump()
}

// -----

// The ENTRY work_e of work does not own the saved local of work. Lowering
// does not mark work_e non_recursive; it is marked here so that the test
// checks the owner.
// CHECK-LABEL: Testing : "entry_owner"
// CHECK: bump_call -> x#0: ModRef{{$}}
module @entry_owner {
  fir.global internal @_QFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QPwork_e() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %0 = fir.address_of(@_QFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPbump()
}

// -----

// The function of the NON_RECURSIVE plain is renamed, and fir.internal_name
// keeps its Fortran name.
// CHECK-LABEL: Testing : "renamed_owner"
// CHECK: bump_call -> w#0: NoModRef{{$}}
module @renamed_owner {
  fir.global internal @_QFplainEw : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @plain_() attributes {fir.internal_name = "_QPplain", fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %0 = fir.address_of(@_QFplainEw) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFplainEw") {test.ptr = "w"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @bump_() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @bump_() attributes {fir.internal_name = "_QPbump"}
}

// -----

// A saved local of a BLOCK construct is owned by the procedure containing the
// BLOCK.
// CHECK-LABEL: Testing : "block_owner"
// CHECK: bump_call -> y#0: NoModRef{{$}}
module @block_owner {
  fir.global internal @_QFworkB1Ey : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QPwork() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %0 = fir.address_of(@_QFworkB1Ey) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFworkB1Ey") {test.ptr = "y"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPbump()
}

// -----

// A saved local of the NON_RECURSIVE internal procedure inner of host.
// CHECK-LABEL: Testing : "internal_owner"
// CHECK: bump_call -> v#0: NoModRef{{$}}
module @internal_owner {
  fir.global internal @_QFhostFinnerEv : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func private @_QFhostPinner() attributes {fir.host_symbol = @_QPhost, fir.proc_attrs = #fir.proc_attrs<non_recursive>, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.address_of(@_QFhostFinnerEv) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFhostFinnerEv") {test.ptr = "v"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPhost()
  func.func private @_QPbump()
}

// -----

// The internal procedure inner of another host does not own the saved local
// of host's inner.
// CHECK-LABEL: Testing : "other_host"
// CHECK: bump_call -> v#0: ModRef{{$}}
module @other_host {
  fir.global internal @_QFhostFinnerEv : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func private @_QFotherPinner() attributes {fir.host_symbol = @_QPother, fir.proc_attrs = #fir.proc_attrs<non_recursive>, llvm.linkage = #llvm.linkage<internal>} {
    %0 = fir.address_of(@_QFhostFinnerEv) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFhostFinnerEv") {test.ptr = "v"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPother()
  func.func private @_QPbump()
}

// -----

// A variable of the main program, which has a single instance.
// CHECK-LABEL: Testing : "main_program"
// CHECK: bump_call -> x#0: NoModRef{{$}}
module @main_program {
  fir.global internal @_QFEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QQmain() attributes {fir.bindc_name = "P"} {
    %0 = fir.address_of(@_QFEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPbump()
}

// -----

// The NON_RECURSIVE work is inlined into the main program, and bump may invoke
// work.
// CHECK-LABEL: Testing : "inlined_into_main_program"
// CHECK: bump_call -> x#0: ModRef{{$}}
module @inlined_into_main_program {
  fir.global internal @_QFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QQmain() attributes {fir.bindc_name = "P"} {
    %0 = fir.address_of(@_QFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPwork() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>}
  func.func private @_QPbump()
}

// -----

// The main program is inlined into another function.
// CHECK-LABEL: Testing : "inlined_main_program"
// CHECK: bump_call -> x#0: ModRef{{$}}
module @inlined_main_program {
  fir.global internal @_QFEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %c0 = arith.constant 0 : i32
    %0 = fir.address_of(@_QFEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QFEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return %c0 : i32
  }
  func.func private @_QPbump()
}

// -----

// The function of a separate module procedure is named after the module that
// declares its interface, and its saved locals after the submodule that
// defines it, so it is not known to own them.
// CHECK-LABEL: Testing : "separate_module_procedure"
// CHECK: bump_call -> x#0: ModRef{{$}}
module @separate_module_procedure {
  fir.global internal @_QMmSsFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QMmPwork() attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %0 = fir.address_of(@_QMmSsFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QMmSsFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    fir.call @_QPbump() {test.ptr = "bump_call"} : () -> ()
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QPbump()
}

// -----

// A call whose nearest function-like operation is not a func.func.
// CHECK-LABEL: Testing : "gpu_kernel"
// CHECK: bump_call -> x#0: ModRef{{$}}
module @gpu_kernel attributes {gpu.container_module} {
  gpu.module @cuda_device_mod {
    fir.global internal @_QMmFworkEx : i32 {
      %0 = fir.zero_bits i32
      fir.has_value %0 : i32
    }
    gpu.func @_QMmPwork() kernel attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
      %0 = fir.address_of(@_QMmFworkEx) : !fir.ref<i32>
      %1 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
      fir.call @_QMmPbump() {test.ptr = "bump_call"} : () -> ()
      %2 = fir.load %1 : !fir.ref<i32>
      gpu.return
    }
    func.func private @_QMmPbump()
  }
}

// -----

// Calls in regions of the NON_RECURSIVE work are code of work, so they cannot
// invoke work either, including a deferred task that runs after work
// returns.
// CHECK-LABEL: Testing : "owner_regions"
// CHECK: task_call -> x#0: NoModRef{{$}}
// CHECK: parallel_call -> x#0: NoModRef{{$}}
// CHECK: loop_call -> x#0: NoModRef{{$}}
module @owner_regions {
  fir.global internal @_QMmFworkEx : i32 {
    %0 = fir.zero_bits i32
    fir.has_value %0 : i32
  }
  func.func @_QMmPwork(%n: index) attributes {fir.proc_attrs = #fir.proc_attrs<non_recursive>} {
    %c1 = arith.constant 1 : index
    %0 = fir.address_of(@_QMmFworkEx) : !fir.ref<i32>
    %1 = fir.declare %0 uniq_name("_QMmFworkEx") {test.ptr = "x"} : (!fir.ref<i32>) -> !fir.ref<i32>
    omp.task {
      fir.call @_QMmPbump() {test.ptr = "task_call"} : () -> ()
      omp.terminator
    }
    omp.parallel {
      fir.call @_QMmPbump() {test.ptr = "parallel_call"} : () -> ()
      omp.terminator
    }
    fir.do_loop %i = %c1 to %n step %c1 {
      fir.call @_QMmPbump() {test.ptr = "loop_call"} : () -> ()
    }
    %2 = fir.load %1 : !fir.ref<i32>
    return
  }
  func.func private @_QMmPbump()
}
