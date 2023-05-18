// RUN: touch %t.S

// RUN: %clang -target x86_64-apple-ios13.1-macabi -darwin-target-variant x86_64-apple-macos10.15 -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.S  2>&1 \
// RUN:   | FileCheck %s

// RUN: %clang -target x86_64-apple-ios-macabi -mmacos-version-min=10.15 -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.S 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SDK-INFO %s

// RUN: %clang -target x86_64-apple-ios-macabi -mmacos-version-min=10.15 -darwin-target-variant x86_64-apple-macos -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.S 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VARIANT-SDK-INFO %s

// RUN: %clang -target x86_64-apple-macos -mmacos-version-min=10.15 -darwin-target-variant x86_64-apple-ios-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -mlinker-version=520 -### %t.S 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-VARIANT-SDK-INFO-INV %s

// CHECK: "-cc1as"
// CHECK-SAME: "-triple" "x86_64-apple-ios13.1.0-macabi"
// CHECK-SAME: "-darwin-target-variant-triple" "x86_64-apple-macos10.15"

// CHECK-SDK-INFO: "-cc1as"
// CHECK-SDK-INFO-SAME: "-triple" "x86_64-apple-ios13.1.0-macabi"
// CHECK-SDK-INFO-SAME: "-target-sdk-version=13.1"

// CHECK-VARIANT-SDK-INFO: "-cc1as"
// CHECK-VARIANT-SDK-INFO-SAME: "-triple" "x86_64-apple-ios13.1.0-macabi"
// CHECK-VARIANT-SDK-INFO-SAME: "-darwin-target-variant-triple" "x86_64-apple-macos"
// CHECK-VARIANT-SDK-INFO-SAME: "-target-sdk-version=13.1"
// CHECK-VARIANT-SDK-INFO-SAME: "-darwin-target-variant-sdk-version=10.15"

// CHECK-VARIANT-SDK-INFO-INV: "-cc1as"
// CHECK-VARIANT-SDK-INFO-INV-SAME: "-triple" "x86_64-apple-macosx10.15.0"
// CHECK-VARIANT-SDK-INFO-INV-SAME: "-darwin-target-variant-triple" "x86_64-apple-ios-macabi"
// CHECK-VARIANT-SDK-INFO-INV-SAME: "-target-sdk-version=10.15"
// CHECK-VARIANT-SDK-INFO-INV-SAME: "-darwin-target-variant-sdk-version=13.1"
