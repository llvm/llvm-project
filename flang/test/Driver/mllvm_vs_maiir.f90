! Verify that `-mllvm` options are forwarded to LLVM and `-maiir` to AIIR.

! In practice, '-maiir --help' is a super-set of '-mllvm --help' and that limits what we can test here. With a better separation of
! LLVM, AIIR and Flang global options, we should be able to write a stricter test.

! RUN: %flang_fc1  -maiir --help | FileCheck %s --check-prefix=AIIR
! RUN: %flang_fc1  -mllvm --help | FileCheck %s --check-prefix=MLLVM

! AIIR: flang (AIIR option parsing) [options]
! AIIR: --aiir-{{.*}}

! MLLVM: flang (LLVM option parsing) [options]
! MLLVM-NOT: --aiir-{{.*}}
