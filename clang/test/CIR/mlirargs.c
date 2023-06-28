// Clang returns 1 when wrong arguments are given.
// RUN: not %clang_cc1 -mmlir -mlir-disable-threadingd 2>&1 | FileCheck %s --check-prefix=WRONG
// Test that the driver can pass mlir args to cc1.
// RUN: %clang -### -mmlir -mlir-disable-threading %s 2>&1 | FileCheck %s --check-prefix=CC1


// WRONG: clang (MLIR option parsing): Unknown command line argument '-mlir-disable-threadingd'.  Try: 'clang (MLIR option parsing) --help'
// WRONG: clang (MLIR option parsing): Did you mean '--mlir-disable-threading'?

// CC1: "-mmlir" "-mlir-disable-threading"
