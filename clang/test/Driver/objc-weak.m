// Check miscellaneous Objective-C options.

// RUN: %clang -target x86_64-apple-macosx10.7 -S -### %s -fobjc-arc -fobjc-weak 2>&1 | FileCheck %s --check-prefix ARC-WEAK
// RUN: %clang -target x86_64-apple-macosx10.7 -S -### %s -fno-objc-weak -fobjc-weak -fobjc-arc  2>&1 | FileCheck %s --check-prefix ARC-WEAK
// ARC-WEAK: -fobjc-arc
// ARC-WEAK: -fobjc-weak

// RUN: %clang -target x86_64-apple-macos10.7 -S -### %s -fobjc-arc -fno-objc-weak 2>&1 | FileCheck %s --check-prefix ARC-NO-WEAK
// RUN: %clang -target x86_64-apple-macos10.7 -S -### %s -fobjc-weak -fno-objc-weak -fobjc-arc  2>&1 | FileCheck %s --check-prefix ARC-NO-WEAK
// ARC-NO-WEAK: -fobjc-arc
// ARC-NO-WEAK: -fno-objc-weak

// RUN: not %clang -target x86_64-apple-macosx10.5 -S -### %s -fobjc-arc -fobjc-weak 2>&1 | FileCheck %s --check-prefix ARC-WEAK-NOTSUPPORTED
// RUN: not %clang -target x86_64-apple-macosx10.5 -S -### %s -fno-objc-weak -fobjc-weak -fobjc-arc  2>&1 | FileCheck %s --check-prefix ARC-WEAK-NOTSUPPORTED
// ARC-WEAK-NOTSUPPORTED: error: -fobjc-weak is not supported on the current deployment target

// RUN: %clang -target x86_64-apple-macos10.7 -S -### %s -fobjc-weak 2>&1 | FileCheck %s --check-prefix MRC-WEAK
// RUN: %clang -target x86_64-apple-macos10.7 -S -### %s -fno-objc-weak -fobjc-weak 2>&1 | FileCheck %s --check-prefix MRC-WEAK
// MRC-WEAK: -fobjc-weak

// RUN: %clang -target x86_64-apple-macosx10.7 -S -### %s -fno-objc-weak 2>&1 | FileCheck %s --check-prefix MRC-NO-WEAK
// RUN: %clang -target x86_64-apple-macosx10.7 -S -### %s -fobjc-weak -fno-objc-weak 2>&1 | FileCheck %s --check-prefix MRC-NO-WEAK
// MRC-NO-WEAK: -fno-objc-weak

// RUN: not %clang -target x86_64-apple-macosx10.5 -S -### %s -fobjc-weak 2>&1 | FileCheck %s --check-prefix MRC-WEAK-NOTSUPPORTED
// RUN: not %clang -target x86_64-apple-macosx10.5 -S -### %s -fno-objc-weak -fobjc-weak 2>&1 | FileCheck %s --check-prefix MRC-WEAK-NOTSUPPORTED
// MRC-WEAK-NOTSUPPORTED: error: -fobjc-weak is not supported on the current deployment target
