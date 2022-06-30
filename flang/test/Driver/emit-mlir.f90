! Test the `-emit-mlir` option

! RUN: %flang_fc1 -emit-mlir %s -o - | FileCheck %s
! RUN: %flang_fc1 -emit-fir %s -o - | FileCheck %s

! Verify that an `.mlir` file is created when `-emit-mlir` is used. Do it in a temporary directory, which will be cleaned up by the
! LIT runner.
! RUN: rm -rf %t-dir && mkdir -p %t-dir && cd %t-dir
! RUN: cp %s .
! RUN: %flang_fc1 -emit-mlir emit-mlir.f90 && ls emit-mlir.mlir

! CHECK: module attributes {
! CHECK-LABEL: func @_QQmain() {
! CHECK-NEXT:  return
! CHECK-NEXT: }
! CHECK-NEXT: }

end program
