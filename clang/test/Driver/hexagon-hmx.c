// Test the hmx flag.

// RUN: %clang -target hexagon -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang -target hexagon -### -mhmx %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-HMX

// RUN: %clang -target hexagon -### -mhmx -mno-hmx %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-HMX

// RUN: %clang -c %s -### -target hexagon -mhmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-HMX %s

// RUN: %clang -c %s -### -target hexagon -mno-hmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-NO-HMX %s
//
// RUN: %clang -target hexagon --help 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-HELP

// CHECK-DEFAULT-NOT: "-target-feature" "+hmx"

// CHECK-HMX: "-target-feature" "+hmx"

// CHECK-NO-HMX: "-target-feature" "-hmx"
// CHECK-NO-HMX-NOT: "-target-feature" "+hmx"

// CHECK-NOIAS-HMX: llvm-mc{{.*}}-mhmx
// CHECK-NOIAS-NO-HMX-NOT: -mhmx

// CHECK-HELP: hmx
// CHECK-HELP: HMX
