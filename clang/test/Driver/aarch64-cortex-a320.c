// RUN: not %clang --target=arm-arm-none-eabi -mcpu=cortex-a320 %s 2>&1 | FileCheck %s
// CHECK: error: unsupported argument {{.*}} to option '-mcpu='

// RUN: %clang -target aarch64 -mcpu=cortex-a320 -### -c %s 2>&1 | FileCheck -check-prefix=A320 %s
// A320: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a320"

