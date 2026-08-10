// RUN: not %clang_cc1 -S -emit-llvm -main-file-name %s 2>&1 | FileCheck %s --check-prefix=ERR1 --implicit-check-not=error:
// ERR1: error: '-S' action ignored; '-emit-llvm' action specified previously

// RUN: not %clang_cc1 -main-file-name %s -emit-llvm-only -emit-llvm -S 2>&1 | FileCheck %s --check-prefix=ERR2 --implicit-check-not=error:
// ERR2: error: '-emit-llvm-only' action ignored; '-S' action specified previously

// RUN: %clang_cc1 -S -main-file-name %s -emit-llvm -o /dev/null
