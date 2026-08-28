// RUN: fir-opt --boxed-procedure="use-safe-trampoline=true" %s | FileCheck %s

// Test that the --boxed-procedure pass with use-safe-trampoline=true
// generates calls to _FortranATrampolineInit, _FortranATrampolineAdjust,
// and _FortranATrampolineFree instead of llvm.init.trampoline and
// llvm.adjust.trampoline intrinsics.

// CHECK-LABEL: func.func @_QPtest_proc_dummy()
// CHECK:         fir.zero_bits !fir.ref<i8>
// CHECK:         fir.convert {{.*}} : {{.*}} -> !fir.ref<i8>
// CHECK:         fir.convert {{.*}} : {{.*}} -> !fir.ref<i8>
// CHECK:         fir.convert {{.*}} : {{.*}} -> !fir.llvm_ptr<i8>
// CHECK:         fir.convert {{.*}} : {{.*}} -> !fir.llvm_ptr<i8>
// CHECK:         fir.convert {{.*}} : {{.*}} -> !fir.llvm_ptr<i8>
// CHECK:         %[[HANDLE:.*]] = fir.call @_FortranATrampolineInit({{.*}}) : (!fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>, !fir.llvm_ptr<i8>) -> !fir.llvm_ptr<i8>
// CHECK:         %[[ADDR:.*]] = fir.call @_FortranATrampolineAdjust(%[[HANDLE]]) : (!fir.llvm_ptr<i8>) -> !fir.llvm_ptr<i8>
// CHECK:         %[[FPTR:.*]] = fir.convert %[[ADDR]]
// CHECK:         fir.call @_QPtest_proc_dummy_other(%[[FPTR]])
// CHECK:         fir.call @_FortranATrampolineFree(%[[HANDLE]])
// CHECK:         return

func.func @_QPtest_proc_dummy() {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c-1_i32 = arith.constant -1 : i32
  %c5_i32 = arith.constant 5 : i32
  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFtest_proc_dummyEi"}
  %1 = fir.alloca tuple<!fir.ref<i32>>
  %2 = fir.coordinate_of %1, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  fir.store %0 to %2 : !fir.llvm_ptr<!fir.ref<i32>>
  fir.store %c1_i32 to %0 : !fir.ref<i32>
  %3 = fir.address_of(@_QFtest_proc_dummyPtest_proc_dummy_a) : (!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> ()
  %4 = fir.emboxproc %3, %1 : ((!fir.ref<i32>, !fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
  fir.call @_QPtest_proc_dummy_other(%4) : (!fir.boxproc<() -> ()>) -> ()
  %5 = fir.address_of(@_QQclX2E2F682E66393000) : !fir.ref<!fir.char<1,8>>
  %6 = fir.convert %5 : (!fir.ref<!fir.char<1,8>>) -> !fir.ref<i8>
  %7 = fir.call @_FortranAioBeginExternalListOutput(%c-1_i32, %6, %c5_i32) : (i32, !fir.ref<i8>, i32) -> !fir.ref<i8>
  %8 = fir.load %0 : !fir.ref<i32>
  %9 = fir.call @_FortranAioOutputInteger32(%7, %8) : (!fir.ref<i8>, i32) -> i1
  %10 = fir.call @_FortranAioEndIoStatement(%7) : (!fir.ref<i8>) -> i32
  return
}
func.func @_QFtest_proc_dummyPtest_proc_dummy_a(%arg0: !fir.ref<i32> {fir.bindc_name = "j"}, %arg1: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.coordinate_of %arg1, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  %1 = fir.load %0 : !fir.llvm_ptr<!fir.ref<i32>>
  %2 = fir.load %1 : !fir.ref<i32>
  %3 = fir.load %arg0 : !fir.ref<i32>
  %4 = arith.addi %2, %3 : i32
  fir.store %4 to %1 : !fir.ref<i32>
  return
}
func.func @_QPtest_proc_dummy_other(%arg0: !fir.boxproc<() -> ()>) {
  %c4_i32 = arith.constant 4 : i32
  %0 = fir.alloca i32 {adapt.valuebyref}
  fir.store %c4_i32 to %0 : !fir.ref<i32>
  %1 = fir.box_addr %arg0 : (!fir.boxproc<() -> ()>) -> ((!fir.ref<i32>) -> ())
  fir.call %1(%0) : (!fir.ref<i32>) -> ()
  return
}
fir.global linkonce @_QQclX2E2F682E66393000 constant : !fir.char<1,8> {
  %0 = fir.string_lit "./h.f90\00"(8) : !fir.char<1,8>
  fir.has_value %0 : !fir.char<1,8>
}
func.func private @_FortranAioOutputInteger32(!fir.ref<i8>, i32) -> i1 attributes {fir.io, fir.runtime}
func.func private @_FortranAioBeginExternalListOutput(i32, !fir.ref<i8>, i32) -> !fir.ref<i8> attributes {fir.io, fir.runtime}
func.func private @_FortranAioEndIoStatement(!fir.ref<i8>) -> i32 attributes {fir.io, fir.runtime}

// Test that Init/Adjust are hoisted to the entry block when emboxproc
// is inside a control flow branch, ensuring the handle dominates
// the TrampolineFree at func.return.
// CHECK-LABEL: func.func @_QPtest_branch
// Init and Adjust should appear before the branch (hoisted to entry block).
// CHECK:         fir.call @_FortranATrampolineInit
// CHECK:         fir.call @_FortranATrampolineAdjust
// CHECK:         cf.cond_br
// Free should appear before return.
// CHECK:         fir.call @_FortranATrampolineFree
// CHECK:         return

func.func @_QPtest_branch(%arg0: i1) {
  %c0_i32 = arith.constant 0 : i32
  %0 = fir.alloca tuple<!fir.ref<i32>>
  %1 = fir.alloca i32 {bindc_name = "x"}
  %2 = fir.coordinate_of %0, %c0_i32 : (!fir.ref<tuple<!fir.ref<i32>>>, i32) -> !fir.llvm_ptr<!fir.ref<i32>>
  fir.store %1 to %2 : !fir.llvm_ptr<!fir.ref<i32>>
  %3 = fir.address_of(@_QPtest_branchPinternal) : (!fir.ref<tuple<!fir.ref<i32>>>) -> ()
  cf.cond_br %arg0, ^bb1, ^bb2
^bb1:
  %4 = fir.emboxproc %3, %0 : ((!fir.ref<tuple<!fir.ref<i32>>>) -> (), !fir.ref<tuple<!fir.ref<i32>>>) -> !fir.boxproc<() -> ()>
  fir.call @_QPdo_something(%4) : (!fir.boxproc<() -> ()>) -> ()
  cf.br ^bb2
^bb2:
  return
}
func.func @_QPtest_branchPinternal(%arg0: !fir.ref<tuple<!fir.ref<i32>>> {fir.host_assoc}) {
  return
}
func.func private @_QPdo_something(!fir.boxproc<() -> ()>)
