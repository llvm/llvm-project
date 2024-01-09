// RUN: %clang --target=aarch64-linux-eabi %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s
// RUN: %clang --target=aarch64-linux-eabi -mfix-cortex-a53-835769 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-YES %s
// RUN: %clang --target=aarch64-linux-eabi -mno-fix-cortex-a53-835769 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NO %s

// RUN: %clang --target=aarch64-linux-androideabi %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-YES %s

// RUN: %clang --target=aarch64-fuchsia %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-YES %s

// RUN: %clang --target=aarch64-fuchsia -mcpu=cortex-a73 %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DEF %s

// CHECK-DEF-NOT: "{[+-]}fix-cortex-a53-835769"
// CHECK-YES: "+fix-cortex-a53-835769"
// CHECK-NO: "-fix-cortex-a53-835769"
