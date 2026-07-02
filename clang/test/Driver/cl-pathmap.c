// Note: %s and %S must be preceded by --, otherwise it may be interpreted as a
// command-line option, e.g. on Mac where %s is commonly under /Users.

// REQUIRES: x86-registered-target
// RUN: %clang_cl --target=x86_64-windows-msvc -### /pathmap:%p=. -- %s 2>&1 | FileCheck %s --check-prefix CHECK-CC1-OPTS
// CHECK-CC1-OPTS: -fdebug-prefix-map=
// CHECK-CC1-OPTS: -fmacro-prefix-map=
// CHECK-CC1-OPTS: -fcoverage-prefix-map=
