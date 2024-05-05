// RUN: not %clang_cc1 -S -emit-llvm -main-file-name %s 2>&1 | FileCheck %s --check-prefix=ERR1 --implicit-check-not=error:
// ERR1: error: action -emit-llvm is specified, another action is not allowed: -S

// RUN: not %clang_cc1 -main-file-name %s -emit-llvm-only -emit-llvm -S 2>&1 | FileCheck %s --check-prefix=ERR2 --implicit-check-not=error:
// ERR2: error: action -S is specified, another action is not allowed: -emit-llvm-only

// RUN: %clang_cc1 -S -main-file-name %s -emit-llvm -o /dev/null
