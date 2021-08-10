// RUN: %clang -target x86_64-apple-ios13.1-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-ios13.99-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c -### %s 2>&1 \
// RUN:   | FileCheck %s

// CHECK: "-target-sdk-version=13.1"
