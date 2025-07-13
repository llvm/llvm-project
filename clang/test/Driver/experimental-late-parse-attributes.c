// RUN: %clang %s -c -fexperimental-late-parse-attributes 2>&1 -### | FileCheck %s -check-prefix=CHECK-ON
// RUN: %clang %s -c -fno-experimental-late-parse-attributes -fexperimental-late-parse-attributes 2>&1 -### | FileCheck %s -check-prefix=CHECK-ON

// CHECK-ON: -cc1
// CHECK-ON: -fexperimental-late-parse-attributes

// RUN: %clang %s -c 2>&1 -### | FileCheck %s -check-prefix=CHECK-OFF
// RUN: %clang %s -c -fno-experimental-late-parse-attributes 2>&1 -### | FileCheck %s -check-prefix=CHECK-OFF
// RUN: %clang %s -c -fexperimental-late-parse-attributes -fno-experimental-late-parse-attributes 2>&1 -### | FileCheck %s -check-prefix=CHECK-OFF

// CHECK-OFF: -cc1
// CHECK-OFF-NOT: -fexperimental-late-parse-attributes
