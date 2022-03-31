// RUN: %clang -target unknown-apple-macos10.15 -arch x86_64 -arch x86_64h -arch i386 \
// RUN:   -darwin-target-variant x86_64-apple-ios13.1-macabi -darwin-target-variant x86_64h-apple-ios13.1-macabi \
// RUN:   -c %s -### 2>&1 | FileCheck %s

// RUN: %clang -target x86_64-apple-macos10.15 -darwin-target-variant i386-apple-ios13.1-macabi \
// RUN:   -c %s -### 2>&1 | FileCheck --check-prefix=UNUSED-TV %s

// RUN: %clang -target x86_64-apple-macos10.15 -darwin-target-variant x86_64-apple-ios13.1-macabi \
// RUN:   -darwin-target-variant x86_64-apple-ios13.1-macabi -c %s -### 2>&1 | FileCheck --check-prefix=REDUNDANT-TV %s

// RUN: %clang -target x86_64-apple-macos10.15 -darwin-target-variant x86_64-apple-ios13.1 \
// RUN:   -c %s -### 2>&1 | FileCheck --check-prefix=INCORRECT-TV %s

// RUN: %clang -target unknown-apple-ios13.1-macabi -arch x86_64 -arch x86_64h \
// RUN:   -darwin-target-variant x86_64-apple-macos10.15 \
// RUN:   -c %s -### 2>&1 | FileCheck --check-prefix=INVERTED %s

// CHECK: "-triple" "x86_64-apple-macosx10.15.0"
// CHECK-SAME: "-darwin-target-variant-triple" "x86_64-apple-ios13.1-macabi"
// CHECK: "-triple" "x86_64h-apple-macosx10.15.0"
// CHECK-SAME: "-darwin-target-variant-triple" "x86_64h-apple-ios13.1-macabi"
// CHECK: "-triple" "i386-apple-macosx10.15.0"
// CHECK-NOT: target-variant-triple

// INVERTED: "-triple" "x86_64-apple-ios13.1.0-macabi"
// INVERTED-SAME: "-darwin-target-variant-triple" "x86_64-apple-macos10.15"
// INVERTED: "-triple" "x86_64h-apple-ios13.1.0-macabi"
// INVERTED-NOT: target-variant-triple

// UNUSED-TV: argument unused during compilation: '-darwin-target-variant i386-apple-ios13.1-macabi'
// REDUNDANT-TV: argument unused during compilation: '-darwin-target-variant x86_64-apple-ios13.1-macabi'
// INCORRECT-TV: unsupported '-darwin-target-variant' value 'x86_64-apple-ios13.1'; use 'ios-macabi' instead
