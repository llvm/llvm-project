// RUN: %clang -target x86_64-apple-ios13.2-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-ios13.2.0-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck %s
// RUN: %clang -target x86_64-apple-ios13.2-macabi -isysroot %S/Inputs/MacOSX10.14.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=FALLBACK-DEFAULT %s
// RUN: %clang -target x86_64-apple-ios12.99.99-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=FALLBACK-DEFAULT %s
// RUN: %clang -target x86_64-apple-ios-macabi -isysroot %S/Inputs/MacOSX10.15.versioned.sdk -c %s -### 2>&1 \
// RUN:   | FileCheck --check-prefix=FALLBACK-DEFAULT %s

// CHECK: -fobjc-runtime=macosx-10.15.1
// FALLBACK-DEFAULT: -fobjc-runtime=macosx-10.15
