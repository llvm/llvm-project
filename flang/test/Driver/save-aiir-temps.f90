! Tests for the `-save-temps` flag. Instead of checking the commands generated
! by the driver with `-###` (like the save-temps.f90 test does), here we check
! that the AIIR files are actually produced in the specified location because
! the driver does not generate specific passes for AIIR. Instead, they are
! generated during code generation as additional outputs.

! As `flang` does not implement `-fc1as` (i.e. a driver for the integrated
! assembler), we need to use `-fno-integrated-as` here.
! However, calling an external assembler on arm64 Macs fails, because it's
! currently being invoked with the `-Q` flag, that is not supported on arm64.
! UNSUPPORTED: system-windows, system-darwin

!--------------------------
! Invalid output directory
!--------------------------
! RUN: not %flang_fc1 -emit-llvm-bc -save-temps=#invalid-dir -o - %s 2>&1 | FileCheck %s -check-prefix=AIIR-ERROR
! AIIR-ERROR: error: Saving AIIR temp file failed

!--------------------------
! Save to cwd
!--------------------------
! RUN: rm -rf %t && mkdir -p %t
! RUN: pushd %t && %flang -c -fno-integrated-as -save-temps=cwd -o out.o %s 2>&1
! RUN: FileCheck %s -input-file=save-aiir-temps-fir.aiir -check-prefix=AIIR-FIR
! RUN: FileCheck %s -input-file=save-aiir-temps-llvmir.aiir -check-prefix=AIIR-LLVMIR
! RUN: popd

! RUN: rm -rf %t && mkdir -p %t
! RUN: pushd %t && %flang -c -fno-integrated-as -save-temps -o out.o %s 2>&1
! RUN: FileCheck %s -input-file=save-aiir-temps-fir.aiir -check-prefix=AIIR-FIR
! RUN: FileCheck %s -input-file=save-aiir-temps-llvmir.aiir -check-prefix=AIIR-LLVMIR
! RUN: popd

!--------------------------
! Save to output directory
!--------------------------
! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang -c -fno-integrated-as -save-temps=obj -o %t/out.o %s 2>&1
! RUN: FileCheck %s -input-file=%t/save-aiir-temps-fir.aiir -check-prefix=AIIR-FIR
! RUN: FileCheck %s -input-file=%t/save-aiir-temps-llvmir.aiir -check-prefix=AIIR-LLVMIR

!--------------------------
! Save to specific directory
!--------------------------
! RUN: rm -rf %t && mkdir -p %t
! RUN: %flang -c -fno-integrated-as -save-temps=%t -o %t/out.o %s 2>&1
! RUN: FileCheck %s -input-file=%t/save-aiir-temps-fir.aiir -check-prefix=AIIR-FIR
! RUN: FileCheck %s -input-file=%t/save-aiir-temps-llvmir.aiir -check-prefix=AIIR-LLVMIR

!--------------------------
! Content to check from the AIIR outputs
!--------------------------
! AIIR-FIR-NOT: llvm.func
! AIIR-FIR: func.func @{{.*}}main(){{.*}}

! AIIR-LLVMIR-NOT: func.func
! AIIR-LLVMIR: llvm.func @{{.*}}main(){{.*}}

end program
