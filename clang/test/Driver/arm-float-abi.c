// RUN: not %clang %s -target armv7-apple-ios -mfloat-abi=hard 2>&1 | FileCheck -check-prefix=ARMV7-ERROR %s
// RUN: %clang %s -target armv7-apple-ios -mfloat-abi=softfp -### 2>&1 | FileCheck -check-prefix=NOERROR %s
// RUN: %clang %s -arch armv7 -target thumbv7-apple-darwin-eabi -mfloat-abi=hard -### 2>&1 | FileCheck -check-prefix=NOERROR %s

// ARMV7-ERROR: unsupported option '-mfloat-abi=hard' for target 'thumbv7'
// NOERROR-NOT: unsupported option

// RUN: %clang -target armv7-linux-androideabi21 %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM7-ANDROID %s
// CHECK-ARM7-ANDROID-NOT: "-target-feature" "+soft-float"
// CHECK-ARM7-ANDROID: "-target-feature" "+soft-float-abi"

// RUN: %clang -target armv8-linux-androideabi21 %s -### -c 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-ARM8-ANDROID %s
// CHECK-ARM8-ANDROID-NOT: "-target-feature" "+soft-float"
// CHECK-ARM8-ANDROID: "-target-feature" "+soft-float-abi"
