// Clang returns 1 when wrong arguments are given.
// RUN: not %clang_cc1 -maiir -aiir-disable-threadingd  -maiir -aiir-print-op-genericd 2>&1 | FileCheck %s --check-prefix=WRONG
// Test that the driver can pass aiir args to cc1.
// RUN: %clang -### -maiir -aiir-disable-threading %s 2>&1 | FileCheck %s --check-prefix=CC1


// WRONG: clang (AIIR option parsing): Unknown command line argument '-aiir-disable-threadingd'.  Try: 'clang (AIIR option parsing) --help'
// WRONG: clang (AIIR option parsing): Did you mean '--aiir-disable-threading'?
// WRONG: clang (AIIR option parsing): Unknown command line argument '-aiir-print-op-genericd'.  Try: 'clang (AIIR option parsing) --help'
// WRONG: clang (AIIR option parsing): Did you mean '--aiir-print-op-generic'?

// CC1: "-maiir" "-aiir-disable-threading"
