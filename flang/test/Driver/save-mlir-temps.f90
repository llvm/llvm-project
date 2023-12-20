! Tests for the `-save-temps` flag. Instead of checking the commands generated
! by the driver with `-###` (like the save-temps.f90 test does), here we check
! that the MLIR files are actually produced in the specified location because
! the driver does not generate specific passes for MLIR. Instead, they are
! generated during code generation as additional outputs.

! As `flang` does not implement `-fc1as` (i.e. a driver for the integrated
! assembler), we need to use `-fno-integrated-as` here.
! However, calling an external assembler on arm64 Macs fails, because it's
! currently being invoked with the `-Q` flag, that is not supported on arm64.
! UNSUPPORTED: system-windows, system-darwin

!--------------------------
! Invalid output directory
!--------------------------
! RUN: not %flang_fc1 -emit-llvm-bc -save-temps=#invalid-dir -o - %s 2>&1 | FileCheck %s -check-prefix=MLIR-ERROR
! MLIR-ERROR: error: Saving MLIR temp file failed

!--------------------------
! Save to cwd
!--------------------------
! RUN: rm -rf %t && mkdir -p %t
! RUN: pushd %t && %flang -c -fno-integrated-as -save-temps=cwd -o out.o %s 2>&1
! RUN: FileCheck %s -input-file=save-mlir-temps-fir.mlir -check-prefix=MLIR-FIR
! RUN: FileCheck %s -input-file=save-mlir-temps-llvmir.mlir -check-prefix=MLIR-LLVMIR
! RUN: popd

! RUN: rm -rf %t && mkdir -p %t
! RUN: pushd %t && %flang -c -fno-integrated-as -save-temps -o out.o %s 2>&1
! RUN: FileCheck %s -input-file=save-mlir-temps-fir.mlir -check-prefix=MLIR-FIR
! RUN: FileCheck %s -input-file=save-mlir-temps-llvmir.mlir -check-prefix=MLIR-LLVMIR
! RUN: popd

!--------------------------
! Save to output directory
!--------------------------
! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang -c -fno-integrated-as -save-temps=obj -o %t/out.o %s 2>&1
! RUN: FileCheck %s -input-file=%t/save-mlir-temps-fir.mlir -check-prefix=MLIR-FIR
! RUN: FileCheck %s -input-file=%t/save-mlir-temps-llvmir.mlir -check-prefix=MLIR-LLVMIR

!--------------------------
! Save to specific directory
!--------------------------
! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang -c -fno-integrated-as -save-temps=%t -o %t/out.o %s 2>&1
! RUN: FileCheck %s -input-file=%t/save-mlir-temps-fir.mlir -check-prefix=MLIR-FIR
! RUN: FileCheck %s -input-file=%t/save-mlir-temps-llvmir.mlir -check-prefix=MLIR-LLVMIR

!--------------------------
! Content to check from the MLIR outputs
!--------------------------
! MLIR-FIR-NOT: llvm.func
! MLIR-FIR: func.func @{{.*}}main() {

! MLIR-FIR-NOT: func.func
! MLIR-LLVMIR: llvm.func @{{.*}}main() {

end program
