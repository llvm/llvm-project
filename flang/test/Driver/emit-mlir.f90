! Test the `-emit-mlir` option

! RUN: %flang_fc1 -emit-mlir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Verify that an `.mlir` file is created when `-emit-mlir` is used. Do it in a temporary directory, which will be cleaned up by the
! LIT runner.
! RUN: rm -rf %t-dir && mkdir -p %t-dir && cd %t-dir
! RUN: cp %s .
! RUN: %flang_fc1 -emit-mlir emit-mlir.f90 && ls emit-mlir.mlir

! CHECK: module attributes {
! CHECK-SAME: dlti.dl_spec =
! CHECK-SAME: llvm.data_layout =
! CHECK-LABEL: func @_QQmain() {
! CHECK-NEXT:  return
! CHECK-NEXT: }
! CHECK-NEXT: func.func private @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr)
! CHECK-NEXT: func.func private @_FortranAProgramEndStatement()
! CHECK-NEXT: func.func @main(%arg0: i32, %arg1: !llvm.ptr, %arg2: !llvm.ptr) -> i32 {
! CHECK-NEXT: %c0_i32 = arith.constant 0 : i32
! CHECK-NEXT: %0 = fir.zero_bits !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>
! CHECK-NEXT: fir.call @_FortranAProgramStart(%arg0, %arg1, %arg2, %0) {{.*}} : (i32, !llvm.ptr, !llvm.ptr, !fir.ref<tuple<i32, !fir.ref<!fir.array<0xtuple<!fir.ref<i8>, !fir.ref<i8>>>>>>) 
! CHECK-NEXT: fir.call @_QQmain() fastmath<contract> : () -> ()
! CHECK-NEXT: fir.call @_FortranAProgramEndStatement() {{.*}} : () -> ()
! CHECK-NEXT: return %c0_i32 : i32
! CHECK-NEXT: }
! CHECK-NEXT: }

end program
