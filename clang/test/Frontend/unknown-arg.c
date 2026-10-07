// RUN: not %clang_cc1 %s -E --helium 2>&1 | \
// RUN: FileCheck %s
// RUN: not %clang_cc1 %s -E --hel[ 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN
// RUN: not %clang %s -E -Xclang --hel[ 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN
// RUN: not %clang_cc1 --version 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN-VER
// RUN: not %clang_cc1 %s -E -mllvm -helium 2>&1 | \
// RUN: FileCheck %s --check-prefix=MLLVM

// CHECK: error: unknown argument: '--helium'
// DID-YOU-MEAN: error: unknown argument '--hel['; did you mean '--help'?
// DID-YOU-MEAN-VER: error: unknown argument '--version'; did you mean '-version'?

// An unknown option behind -mllvm is reported, and the text LLVM's own parser
// writes is unchanged, because it still goes to the stream it used before.
// MLLVM: clang (LLVM option parsing): Unknown command line argument '-helium'
// MLLVM: error: invalid argument in '-mllvm'
