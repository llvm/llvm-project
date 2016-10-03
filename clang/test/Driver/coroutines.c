// RUN: %clang -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CORO %s
// RUN: %clang -fcoroutines-ts -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CORO %s
// RUN: %clang -fno-coroutines-ts -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CORO %s
// RUN: %clang -fno-coroutines-ts -fcoroutines-ts -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-CORO %s
// CHECK-NO-CORO-NOT: -fcoroutines-ts

