// RUN: %clang -target i386 %s -fpatchable-function-entry=1 -c -### 2>&1 | FileCheck %s
// RUN: %clang -target x86_64 %s -fpatchable-function-entry=1 -c -### 2>&1 | FileCheck %s
// RUN: %clang -target aarch64 %s -fpatchable-function-entry=1 -c -### 2>&1 | FileCheck %s
// RUN: %clang -target aarch64 %s -fpatchable-function-entry=1,0 -c -### 2>&1 | FileCheck %s
// CHECK: "-fpatchable-function-entry=1"

// RUN: not %clang -target ppc64 -fsyntax-only %s -fpatchable-function-entry=1 2>&1 | FileCheck --check-prefix=TARGET %s
// TARGET: error: unsupported option '-fpatchable-function-entry=1' for target 'ppc64'

// RUN: not %clang -target i386 -fsyntax-only %s -fpatchable-function-entry=1,1 2>&1 | FileCheck --check-prefix=NONZERO %s
// NONZERO: error: the second argument of '-fpatchable-function-entry' must be 0 or omitted

// RUN: not %clang -target x86_64 -fsyntax-only %s -fpatchable-function-entry=1,0, 2>&1 | FileCheck --check-prefix=EXCESS %s
// EXCESS: error: invalid argument '1,0,' to -fpatchable-function-entry=

// RUN: not %clang -target aarch64-linux -fsyntax-only %s -fxray-instrument -fpatchable-function-entry=1 2>&1 | FileCheck --check-prefix=XRAY %s
// XRAY: error: invalid argument '-fxray-instrument' not allowed with '-fpatchable-function-entry='
