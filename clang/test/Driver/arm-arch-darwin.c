// On Darwin, arch should override CPU for triple purposes
// RUN: %clang -target armv7m-apple-darwin -arch armv7m -mcpu=cortex-m4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7M-DARWIN %s
// CHECK-V7M-DARWIN: "-cc1"{{.*}} "-triple" "thumbv7m-{{.*}} "-target-cpu" "cortex-m4"
// RUN: %clang -target armv7m -arch armv7m -mcpu=cortex-m4 -### -c %s 2>&1 | FileCheck -check-prefix=CHECK-V7M-OVERRIDDEN %s
// CHECK-V7M-OVERRIDDEN: "-cc1"{{.*}} "-triple" "thumbv7em-{{.*}} "-target-cpu" "cortex-m4"

