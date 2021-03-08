// arm64 Mac-based targets default to Apple A13.

// RUN: %clang -target arm64-apple-macos             -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-ios-macabi        -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-ios-simulator     -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-watchos-simulator -### -c %s 2>&1 | FileCheck %s
// RUN: %clang -target arm64-apple-tvos-simulator    -### -c %s 2>&1 | FileCheck %s

// RUN: %clang -target arm64-apple-macos -arch arm64 -### -c %s 2>&1 | FileCheck %s

// RUN: %clang -target arm64-apple-macos -mcpu=apple-a11 -### -c %s 2>&1 | FileCheck --check-prefix=EXPLICIT-A11 %s
// RUN: %clang -target arm64-apple-macos -mcpu=apple-a7  -### -c %s 2>&1 | FileCheck --check-prefix=EXPLICIT-A7 %s
// RUN: %clang -target arm64-apple-macos -mcpu=apple-a13 -### -c %s 2>&1 | FileCheck --check-prefix=EXPLICIT-A13 %s

// CHECK: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "apple-a13"
// CHECK-SAME: "-target-feature" "+v8.4a"

// EXPLICIT-A11: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "apple-a11"
// EXPLICIT-A7: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "apple-a7"
// EXPLICIT-A13: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "apple-a13"
