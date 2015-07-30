// Test that different values of -Wa,-mcpu/mfpu/march/mhwdiv pick correct ARM target-feature(s).
// Complete tests about -mcpu/mfpu/march/mhwdiv on other files.

// CHECK-DUP-CPU: warning: argument unused during compilation: '-mcpu=cortex-a8'
// CHECK-DUP-FPU: warning: argument unused during compilation: '-mfpu=vfpv3'
// CHECK-DUP-ARCH: warning: argument unused during compilation: '-march=armv7'
// CHECK-DUP-HDIV: warning: argument unused during compilation: '-mhwdiv=arm'

// CHECK: "cc1as"
// ================================================================= CPU
// RUN: %clang -target arm-linux-gnueabi -Wa,-mcpu=cortex-a15 -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-CPU %s
// CHECK-CPU: "-target-cpu" "cortex-a15"

// RUN: %clang -target arm -Wa,-mcpu=bogus -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-BOGUS-CPU %s
// CHECK-BOGUS-CPU: error: {{.*}} does not support '-Wa,-mcpu=bogus'

// RUN: %clang -target arm -mcpu=cortex-a8 -Wa,-mcpu=cortex-a15 -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DUP-CPU %s
// CHECK-DUP-CPU: "-target-cpu" "cortex-a15"

// ================================================================= FPU
// RUN: %clang -target arm-linux-eabi -Wa,-mfpu=neon -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-NEON %s
// CHECK-NEON: "-target-feature" "+neon"

// RUN: %clang -target arm-linux-eabi -Wa,-mfpu=bogus -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BOGUS-FPU %s
// CHECK-BOGUS-FPU: error: {{.*}} does not support '-Wa,-mfpu=bogus'

// RUN: %clang -target arm-linux-eabi -mfpu=vfpv3 -Wa,-mfpu=neon -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DUP-FPU %s
// CHECK-DUP-FPU: "-target-feature" "+neon"

// ================================================================= Arch
// Arch validation only for now, in case we're passing to an external asm

// RUN: %clang -target arm -Wa,-march=armbogusv6 -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-BOGUS-ARCH %s
// CHECK-BOGUS-ARCH: error: {{.*}} does not support '-Wa,-march=armbogusv6'

// RUN: %clang -target arm -march=armv7 -Wa,-march=armv6 -c %s -### 2>&1 \
// RUN:   | FileCheck -check-prefix=CHECK-DUP-ARCH %s

// ================================================================= HD Div
// RUN: %clang -target arm -Wa,-mhwdiv=arm -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM %s
// CHECK-ARM: "-target-feature" "+hwdiv-arm"
// CHECK-ARM: "-target-feature" "-hwdiv"

// RUN: %clang -target arm -Wa,-mhwdiv=thumb -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-THUMB %s
// CHECK-THUMB: "-target-feature" "-hwdiv-arm"
// CHECK-THUMB: "-target-feature" "+hwdiv"

// RUN: %clang -target arm -Wa,-mhwdiv=bogus -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-BOGUS-HDIV %s
// CHECK-BOGUS-HDIV: error: {{.*}} does not support '-Wa,-mhwdiv=bogus'

// RUN: %clang -target arm -mhwdiv=arm -Wa,-mhwdiv=thumb -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-DUP-HDIV %s
// CHECK-DUP-HDIV: "-target-feature" "-hwdiv-arm"
// CHECK-DUP-HDIV: "-target-feature" "+hwdiv"
