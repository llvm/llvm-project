! Verify that `-mllvm` options are forwarded to LLVM and `-mmlir` to MLIR.

! In practice, '-mmlir --help' is a super-set of '-mllvm --help' and that limits what we can test here. With a better separation of
! LLVM, MLIR and Flang global options, we should be able to write a stricter test.

! RUN: %flang_fc1  -mmlir --help | FileCheck %s --check-prefix=MLIR
! RUN: %flang_fc1  -mllvm --help | FileCheck %s --check-prefix=MLLVM

! MLIR: flang (MLIR option parsing) [options]
! MLIR: --mlir-{{.*}}

! MLLVM: flang (LLVM option parsing) [options]
! MLLVM-NOT: --mlir-{{.*}}
