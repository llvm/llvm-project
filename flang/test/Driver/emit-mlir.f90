! Test the `-emit-mlir` option

! RUN: %flang_fc1 -emit-mlir %s -o - | FileCheck %s

! Verify that an `.mlir` file is created when `-emit-mlir` is used. Do it in a temporary directory, which will be cleaned up by the
! LIT runner.
! RUN: rm -rf %t-dir && mkdir -p %t-dir && cd %t-dir
! RUN: cp %s .
! RUN: %flang_fc1 -emit-mlir emit-mlir.f90 && ls emit-mlir.mlir

! CHECK-LABEL:   llvm.func @_QQmain() {
! CHECK:           llvm.return
! CHECK:         }
! CHECK:         llvm.func @_FortranAProgramStart(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) attributes {sym_visibility = "private"}
! CHECK:         llvm.func @_FortranAProgramEndStatement() attributes {sym_visibility = "private"}

! CHECK-LABEL:   llvm.func @main(
! CHECK-SAME:                    %[[ARG0:.*]]: i32,
! CHECK-SAME:                    %[[ARG1:.*]]: !llvm.ptr,
! CHECK-SAME:                    %[[ARG2:.*]]: !llvm.ptr) -> i32 {
! CHECK:           %[[VAL_0:.*]] = llvm.mlir.constant(0 : i32) : i32
! CHECK:           %[[VAL_1:.*]] = llvm.mlir.zero : !llvm.ptr
! CHECK:           llvm.call @_FortranAProgramStart(%[[ARG0]], %[[ARG1]], %[[ARG2]], %[[VAL_1]]) {fastmathFlags = #llvm.fastmath<contract>} : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
! CHECK:           llvm.call @_QQmain() {fastmathFlags = #llvm.fastmath<contract>} : () -> ()
! CHECK:           llvm.call @_FortranAProgramEndStatement() {fastmathFlags = #llvm.fastmath<contract>} : () -> ()
! CHECK:           llvm.return %[[VAL_0]] : i32
! CHECK:         }

end program
