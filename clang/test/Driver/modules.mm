// RUN: %clang -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// RUN: %clang -fcxx-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// CHECK-NO-MODULES-NOT: -fmodules

// RUN: %clang -fcxx-modules -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// CHECK-HAS-MODULES: -fmodules
