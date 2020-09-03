// Check miscellaneous Objective-C options.

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch x86_64 -fobjc-abi-version=1 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-X86_64_ABI1 < %t %s

// CHECK-CHECK-X86_64_ABI1: "-cc1"
// CHECK-CHECK-X86_64_ABI1: -fobjc-runtime=macosx-fragile-10.6.0
// CHECK-CHECK-X86_64_ABI1-NOT: -fobjc-dispatch-method
// CHECK-CHECK-X86_64_ABI1: darwin-objc-options

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch i386 -fobjc-abi-version=2 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-I386_ABI2 < %t %s

// CHECK-CHECK-I386_ABI2: "-cc1"
// CHECK-CHECK-I386_ABI2: -fobjc-runtime=macosx-10.6.0
// CHECK-CHECK-I386_ABI2: -fobjc-exceptions
// CHECK-CHECK-I386_ABI2: -fexceptions
// CHECK-CHECK-I386_ABI2-NOT: -fobjc-dispatch-method
// CHECK-CHECK-I386_ABI2: darwin-objc-options

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch i386 -fobjc-runtime=ios-5.0 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-I386_IOS < %t %s

// CHECK-CHECK-I386_IOS: "-cc1"
// CHECK-CHECK-I386_IOS: -fobjc-runtime=ios-5.0
// CHECK-CHECK-I386_IOS: -fobjc-exceptions
// CHECK-CHECK-I386_IOS: -fexceptions
// CHECK-CHECK-I386_IOS-NOT: -fobjc-dispatch-method
// CHECK-CHECK-I386_IOS: darwin-objc-options

/// Don't add -fobjc-runtime for non-ObjC input.
// RUN: touch %t.c
// RUN: %clang -target x86_64-apple-darwin -x objective-c -S -### %t.c 2>&1 | FileCheck --check-prefix=F %s
// RUN: %clang -target x86_64-apple-darwin -S -### %t.c 2>&1 | FileCheck --check-prefix=NO_F %s
// F: -fobjc-runtime=
// NO_F-NOT: -fobjc-runtime=

// Don't crash with an unexpected target triple.
// RUN: %clang -target i386-apple-ios7 -S -### %s

// Add -fcompatibility-qualified-id-block-type-checking only on Darwin.
// RUN: %clang -target x86_64-apple-darwin10 -### %s 2>&1 | FileCheck --check-prefix=DARWIN_COMPATIBILITY %s
// RUN: %clang -target x86_64-linux-gnu -### %s 2>&1 | FileCheck --check-prefix=OTHER_COMPATIBILITY %s
// DARWIN_COMPATIBILITY: -fcompatibility-qualified-id-block-type-checking
// OTHER_COMPATIBILITY-NOT: -fcompatibility-qualified-id-block-type-checking
