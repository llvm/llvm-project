// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CORO %s
// RUN: %clang -fcoroutines-ts -fno-coroutines-ts -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CORO %s
// RUN: %clang -fno-coroutines-ts -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CORO %s
// CHECK-NO-CORO-NOT: -fcoroutines-ts

// RUN: %clang -fcoroutines-ts -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-CORO  %s
// RUN: %clang -fno-coroutines-ts -fcoroutines-ts -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-CORO %s
// CHECK-HAS-CORO: the '-fcoroutines-ts' flag is deprecated and it will be removed in Clang 17; use '-std=c++20' or higher to use standard C++ coroutines instead
// CHECK-HAS-CORO: -fcoroutines-ts

