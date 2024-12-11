// Check that darwin passes -fbuiltin-headers-in-system-modules
// when expected.

// RUN: %clang -target x86_64-apple-darwin22.4 -### %s 2>&1 | FileCheck %s
// RUN: %clang -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -target x86_64-apple-macos10.15 -### %s 2>&1 | FileCheck %s
// RUN: %clang -isysroot %S/Inputs/iPhoneOS13.0.sdk -target arm64-apple-ios13.0 -### %s 2>&1 | FileCheck %s
// CHECK: -fbuiltin-headers-in-system-modules

// RUN: %clang -isysroot %S/Inputs/MacOSX15.0.sdk -target x86_64-apple-macos14.0 -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// RUN: %clang -isysroot %S/Inputs/MacOSX15.0.sdk -target x86_64-apple-macos15.0 -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// RUN: %clang -isysroot %S/Inputs/MacOSX15.0.sdk -target x86_64-apple-ios18.0-macabi -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// RUN: %clang -isysroot %S/Inputs/MacOSX15.1.sdk -target x86_64-apple-macos15.1 -darwin-target-variant x86_64-apple-ios18.1-macabi -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// RUN: %clang -isysroot %S/Inputs/MacOSX15.1.sdk -target x86_64-apple-ios18.1-macabi -darwin-target-variant x86_64-apple-macos15.1 -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// RUN: %clang -isysroot %S/Inputs/DriverKit23.0.sdk -target arm64-apple-driverkit23.0 -### %s 2>&1 | FileCheck --check-prefix=CHECK_FUTURE %s
// CHECK_FUTURE-NOT: -fbuiltin-headers-in-system-modules
