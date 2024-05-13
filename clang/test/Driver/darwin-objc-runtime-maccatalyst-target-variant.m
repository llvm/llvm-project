// RUN: %clang -target x86_64-apple-ios13.2-macabi -darwin-target-variant x86_64-apple-macos10.15.3-macos -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-macos10.15.1 -darwin-target-variant x86_64-apple-ios13.2-macabi  -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-ios13.2-macabi -darwin-target-variant x86_64-apple-macos10.15-macos -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=LOWER %s

// CHECK: -fobjc-runtime=macosx-10.15.1
// LOWER: -fobjc-runtime=macosx-10.15
