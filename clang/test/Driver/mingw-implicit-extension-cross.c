// Test how an implicit .exe extension is added. If not running the compiler
// on windows, no implicit extension is added. (Therefore, this test is skipped
// when running on windows.)

// UNSUPPORTED: system-windows

// RUN: %clang -target i686-windows-gnu -### --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -o outputname 2>&1 | FileCheck %s

// CHECK: "-o" "outputname"
