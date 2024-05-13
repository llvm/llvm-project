// RUN: %clang -target x86_64-apple-macosx10.15 -darwin-target-variant x86_64-apple-ios13.1-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: env SDKROOT=%S/Inputs/MacOSX10.15.versioned.sdk %clang -target x86_64-apple-macosx10.15 -darwin-target-variant x86_64-apple-ios13.1-macabi -c -### %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-ios13.1-macabi -darwin-target-variant x86_64-apple-macosx10.15 -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-SWAPPED %s
// RUN: %clang -target x86_64-apple-ios13.1-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck --check-prefix=CHECK-MACCATALYST %s

// CHECK: "-target-sdk-version=10.15" "-darwin-target-variant-sdk-version=13.1"
// CHECK-SWAPPED: "-target-sdk-version=13.1" "-darwin-target-variant-sdk-version=10.15"
// CHECK-MACCATALYST: "-target-sdk-version=13.1"
