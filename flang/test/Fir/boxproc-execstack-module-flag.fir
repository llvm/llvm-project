// RUN: fir-opt --pass-pipeline='builtin.module(builtin.module(boxed-procedure))' -mlir-disable-threading %s | FileCheck %s

// Test how the --boxed-procedure pass adds the "executable-stack" module flag
// requested by a stack based trampoline to the flags a module already carries.
// Running the pass over several nested modules also checks that the request
// does not leak from one run of the pass to the next.

// No flags yet: a new llvm.module_flags op is created.

// CHECK-LABEL: module @no_flags {
// CHECK-NEXT:    llvm.module_flags [#llvm.mlir.module_flag<max, "executable-stack", 1 : i32>]
module @no_flags {
  func.func @_QPtest_proc_dummy() {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.alloca i32
    %1 = fir.alloca tuple<!fir.ref<i32>>
    %2 = fir.coordinate_of %1, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
    fir.store %0 to %2 : !fir.llvm_ptr<!fir.ref<i32>>
    %3 = fir.address_of(@_QFtest_proc_dummyPtest_proc_dummy_a) : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
    %4 = fir.emboxproc %3, %1 : ((!fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
    fir.call @_QPtest_proc_dummy_other(%4) : (!fir.boxproc<() -> ()>) -> ()
    return
  }
  func.func @_QFtest_proc_dummyPtest_proc_dummy_a(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {
    return
  }
  func.func private @_QPtest_proc_dummy_other(!fir.boxproc<() -> ()>)
}

// Unrelated flags are preserved and the new one is appended.

// CHECK-LABEL: module @other_flags {
// CHECK-NEXT:    llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>, #llvm.mlir.module_flag<max, "executable-stack", 1 : i32>]
module @other_flags {
  llvm.module_flags [#llvm.mlir.module_flag<error, "wchar_size", 4 : i32>]
  func.func @_QPtest_proc_dummy() {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.alloca i32
    %1 = fir.alloca tuple<!fir.ref<i32>>
    %2 = fir.coordinate_of %1, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
    fir.store %0 to %2 : !fir.llvm_ptr<!fir.ref<i32>>
    %3 = fir.address_of(@_QFtest_proc_dummyPtest_proc_dummy_a) : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
    %4 = fir.emboxproc %3, %1 : ((!fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
    fir.call @_QPtest_proc_dummy_other(%4) : (!fir.boxproc<() -> ()>) -> ()
    return
  }
  func.func @_QFtest_proc_dummyPtest_proc_dummy_a(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {
    return
  }
  func.func private @_QPtest_proc_dummy_other(!fir.boxproc<() -> ()>)
}

// An existing "executable-stack" is overwritten (0 -> 1), not appended to.

// CHECK-LABEL: module @execstack_zero {
// CHECK-NEXT:    llvm.module_flags [#llvm.mlir.module_flag<max, "executable-stack", 1 : i32>, #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>]
// CHECK-NOT:     "executable-stack"
module @execstack_zero {
  llvm.module_flags [#llvm.mlir.module_flag<max, "executable-stack", 0 : i32>,
                     #llvm.mlir.module_flag<error, "wchar_size", 4 : i32>]
  func.func @_QPtest_proc_dummy() {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.alloca i32
    %1 = fir.alloca tuple<!fir.ref<i32>>
    %2 = fir.coordinate_of %1, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
    fir.store %0 to %2 : !fir.llvm_ptr<!fir.ref<i32>>
    %3 = fir.address_of(@_QFtest_proc_dummyPtest_proc_dummy_a) : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
    %4 = fir.emboxproc %3, %1 : ((!fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
    fir.call @_QPtest_proc_dummy_other(%4) : (!fir.boxproc<() -> ()>) -> ()
    return
  }
  func.func @_QFtest_proc_dummyPtest_proc_dummy_a(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {
    return
  }
  func.func private @_QPtest_proc_dummy_other(!fir.boxproc<() -> ()>)
}

// An entry that already requests an executable stack stays as it is.

// CHECK-LABEL: module @execstack_one {
// CHECK-NEXT:    llvm.module_flags [#llvm.mlir.module_flag<max, "executable-stack", 1 : i32>]
// CHECK-NOT:     "executable-stack"
module @execstack_one {
  llvm.module_flags [#llvm.mlir.module_flag<max, "executable-stack", 1 : i32>]
  func.func @_QPtest_proc_dummy() {
    %c0_i32 = arith.constant 0 : i32
    %0 = fir.alloca i32
    %1 = fir.alloca tuple<!fir.ref<i32>>
    %2 = fir.coordinate_of %1, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
    fir.store %0 to %2 : !fir.llvm_ptr<!fir.ref<i32>>
    %3 = fir.address_of(@_QFtest_proc_dummyPtest_proc_dummy_a) : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
    %4 = fir.emboxproc %3, %1 : ((!fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
    fir.call @_QPtest_proc_dummy_other(%4) : (!fir.boxproc<() -> ()>) -> ()
    return
  }
  func.func @_QFtest_proc_dummyPtest_proc_dummy_a(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {
    return
  }
  func.func private @_QPtest_proc_dummy_other(!fir.boxproc<() -> ()>)
}

// A module with no stack based trampoline gets no flag, even though the
// preceding modules asked for one.

// CHECK-LABEL: module @no_trampoline {
// CHECK-NOT:     llvm.module_flags
module @no_trampoline {
  func.func @_QPtest_no_trampoline(%arg0: !fir.boxproc<() -> ()>) {
    %0 = fir.box_addr %arg0 : (!fir.boxproc<() -> ()>) -> (() -> ())
    fir.call %0() : () -> ()
    return
  }
}
