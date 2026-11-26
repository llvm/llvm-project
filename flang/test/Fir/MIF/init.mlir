// RUN: fir-opt --split-input-file --mif-convert %s | FileCheck %s

  func.func @_QQmain() attributes {fir.bindc_name = "TEST_INIT"} {
    %0 = fir.dummy_scope : !fir.dscope
    return
  }
  func.func private @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
  func.func private @_FortranAProgramEndStatement()
  func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
    %0 = fir.zero_bits !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
    fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0) fastmath<contract> : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>) -> ()
    %1 = mif.init -> i32
    fir.call @_QQmain() fastmath<contract> : () -> ()
    fir.call @_FortranAProgramEndStatement() fastmath<contract> : () -> ()
    %c0_i32 = arith.constant 0 : i32
    return %c0_i32 : i32
  }

// CHECK-LABEL: func.func @main
// CHECK: %[[VAL_0:.*]] = fir.alloca i32
// CHECK: %[[VAL_1:.*]] = fir.address_of(@_QMprifPprif_stop_termination_wrapper) : (i32) -> () 
// CHECK: fir.call @_FortranARegisterImagesNormalEndCallback({{.*}}) : (!fir.llvm_ptr<() -> ()>) -> ()
// CHECK: %[[VAL_2:.*]] = fir.address_of(@_QMprifPprif_error_stop_termination_wrapper) : (i32) -> () 
// CHECK: fir.call @_FortranARegisterImagesErrorCallback({{.*}}) : (!fir.llvm_ptr<() -> ()>) -> ()
// CHECK: %[[VAL_3:.*]] = fir.address_of(@_QMprifPprif_fail_image_termination_wrapper) : () -> ()
// CHECK: fir.call @_FortranARegisterFailImageCallback({{.*}}) : (!fir.llvm_ptr<() -> ()>) -> ()
// CHECK: fir.call @_QMprifPprif_init(%[[VAL_0]]) : (!fir.ref<i32>) -> ()

// CHECK:  func.func private @_QMprifPprif_stop_termination_wrapper(%[[ARG0:.*]]: i32)
// CHECK:  %[[VAL_0:.*]] = fir.alloca i32
// CHECK:  %[[VAL_1:.*]] = fir.alloca i1
// CHECK:  %[[TRUE:.*]] = arith.constant true
// CHECK:  fir.store %[[TRUE]] to %[[VAL_1]] : !fir.ref<i1>
// CHECK:  %[[VAL_2:.*]] = fir.absent !fir.boxchar<1>
// CHECK:  fir.store %[[ARG0]] to %[[VAL_0]] : !fir.ref<i32>
// CHECK:  fir.call @_QMprifPprif_stop(%[[VAL_1]], %[[VAL_0]], %[[VAL_2]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()
// CHECK:  return

// CHECK:  func.func private @_QMprifPprif_error_stop_termination_wrapper(%[[ARG0:.*]]: i32)
// CHECK:  %[[VAL_0:.*]] = fir.alloca i32
// CHECK:  %[[VAL_1:.*]] = fir.alloca i1
// CHECK:  %[[TRUE:.*]] = arith.constant true 
// CHECK:  fir.store %[[TRUE]] to %[[VAL_1]] : !fir.ref<i1>
// CHECK:  %[[VAL_2:.*]] = fir.absent !fir.boxchar<1>
// CHECK:  fir.store %[[ARG0]] to %[[VAL_0]] : !fir.ref<i32>
// CHECK:  fir.call @_QMprifPprif_error_stop(%[[VAL_1]], %[[VAL_0]], %[[VAL_2]]) : (!fir.ref<i1>, !fir.ref<i32>, !fir.boxchar<1>) -> ()
// CHECK:  return

// CHECK-LABEL:  func.func private @_QMprifPprif_fail_image_termination_wrapper
// CHECK:  fir.call @_QMprifPprif_fail_image() : () -> ()
// CHECK:  return

