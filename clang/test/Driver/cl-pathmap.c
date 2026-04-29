// RUN: %clang_cl -E -### /pathmap:%p=. %s 2>&1 | FileCheck %s --check-prefix CHECK-CC1-OPTS
// CHECK-CC1-OPTS: -fdebug-prefix-map=
// CHECK-CC1-OPTS: -fmacro-prefix-map=
// CHECK-CC1-OPTS: -fcoverage-prefix-map=
