// Test the hmx flag.

// RUN: %clang --target=hexagon -### %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-DEFAULT

// RUN: %clang --target=hexagon -### -mhmx %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-HMX

// RUN: %clang --target=hexagon -### -mhmx -mno-hmx %s 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-NO-HMX

// RUN: %clang -c %s -### --target=hexagon -fno-integrated-as 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECK-NOIAS-DEFAULT %s

// RUN: %clang -c %s -### --target=hexagon -mhmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-HMX %s

// RUN: %clang -c %s -### --target=hexagon -mno-hmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-NO-HMX %s

// Normalize the external assembler's feature list: last option wins.
// RUN: %clang -c %s -### --target=hexagon -mno-hmx -mhmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-LAST-HMX %s
// RUN: %clang -c %s -### --target=hexagon -mhmx -mno-hmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-LAST-NO-HMX %s

// Repeated identical options produce only one feature entry.
// RUN: %clang -c %s -### --target=hexagon -mhmx -mhmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-LAST-HMX %s
// RUN: %clang -c %s -### --target=hexagon -mno-hmx -mno-hmx \
// RUN:  -fno-integrated-as 2>&1 | FileCheck \
// RUN:  -check-prefix=CHECK-NOIAS-LAST-NO-HMX %s

// RUN: %clang --target=hexagon --help 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-HELP

// CHECK-DEFAULT-NOT: "-target-feature" "+hmx"

// CHECK-HMX: "-target-feature" "+hmx"

// CHECK-NO-HMX: "-target-feature" "-hmx"
// CHECK-NO-HMX-NOT: "-target-feature" "+hmx"

// CHECK-NOIAS-DEFAULT: llvm-mc
// CHECK-NOIAS-DEFAULT-NOT: "-mattr={{[^"]*[+-]hmx[^"]*}}"
// CHECK-NOIAS-DEFAULT-NOT: "-mhmx"
// CHECK-NOIAS-HMX: llvm-mc{{.*}}"-mattr={{[^"]*}}+hmx{{[^"]*}}"
// CHECK-NOIAS-HMX-NOT: "-mhmx"
// CHECK-NOIAS-NO-HMX: llvm-mc{{.*}}"-mattr={{[^"]*}}-hmx{{[^"]*}}"
// CHECK-NOIAS-NO-HMX-NOT: "-mhmx"
// CHECK-NOIAS-LAST-HMX: llvm-mc{{.*}}"-mattr=+hmx,
// CHECK-NOIAS-LAST-HMX-NOT: {{[+-]}}hmx{{[,"]}}
// CHECK-NOIAS-LAST-HMX-NOT: "-mhmx"
// CHECK-NOIAS-LAST-NO-HMX: llvm-mc{{.*}}"-mattr=-hmx,
// CHECK-NOIAS-LAST-NO-HMX-NOT: {{[+-]}}hmx{{[,"]}}
// CHECK-NOIAS-LAST-NO-HMX-NOT: "-mhmx"

// CHECK-HELP: hmx
// CHECK-HELP: HMX
