// RUN: %clang -fmodules -fno-modules -### %s 2>&1 | FileCheck -check-prefix=CHECK-NO-MODULES %s
// CHECK-NO-MODULES-NOT: -fmodules

// RUN: %clang -fmodules -fno-modules -fmodules -### %s 2>&1 | FileCheck -check-prefix=CHECK-HAS-MODULES %s
// CHECK-HAS-MODULES: -fmodules

