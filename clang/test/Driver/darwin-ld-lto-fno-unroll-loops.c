// REQUIRES: system-darwin

// RUN: mkdir -p %t/bin
// RUN: mkdir -p %t/lib
// RUN: touch %t/lib/libLTO.dylib

// Check that ld gets "-lto-no-unroll-loops" when -fno-unroll-loops is passed.
//
// RUN: %clang -target x86_64-apple-darwin10 %s -fno-unroll-loops -flto=full -### 2>&1 | \
// RUN:   FileCheck --check-prefix=NOUNROLL %s

// NOUNROLL:  "-mllvm" "-lto-no-unroll-loops"
//
// RUN: %clang -target x86_64-apple-darwin10 %s -flto=full -### 2>&1 | \
// RUN:   FileCheck --check-prefix=UNROLL %s

// UNROLL-NOT:  -lto-no-unroll-loops
