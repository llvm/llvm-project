// Test how an implicit .exe extension is added. If running the compiler
// on windows, an implicit extension is added if none is provided in the
// given name. (Therefore, this test is skipped when not running on windows.)

// REQUIRES: system-windows

// RUN: %clang -target i686-windows-gnu -### --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -o outputname 2>&1 | FileCheck %s --check-prefix=CHECK-OUTPUTNAME-EXE

// RUN: %clang -target i686-windows-gnu -### --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -o outputname.exe 2>&1 | FileCheck %s --check-prefix=CHECK-OUTPUTNAME-EXE

// RUN: %clang -target i686-windows-gnu -### --sysroot=%S/Inputs/mingw_clang_tree/mingw32 %s -o outputname.q 2>&1 | FileCheck %s --check-prefix=CHECK-OUTPUTNAME-Q

// CHECK-OUTPUTNAME-EXE: "-o" "outputname.exe"
// CHECK-OUTPUTNAME-Q: "-o" "outputname.q"
