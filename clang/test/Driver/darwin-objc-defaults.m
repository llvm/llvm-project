// Check non-fragile ABI and dispatch method defaults.

// i386

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch i386 -mmacos-version-min=10.5 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-I386_OSX10_5 < %t %s

// CHECK-CHECK-I386_OSX10_5: "-cc1"
// CHECK-CHECK-I386_OSX10_5: -fobjc-runtime=macosx-fragile-10.5
// CHECK-CHECK-I386_OSX10_5-NOT: -fobjc-dispatch-method
// CHECK-CHECK-I386_OSX10_5: darwin-objc-defaults

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch i386 -mmacos-version-min=10.6 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-I386_OSX10_6 < %t %s

// CHECK-CHECK-I386_OSX10_6: "-cc1"
// CHECK-CHECK-I386_OSX10_6: -fobjc-runtime=macosx-fragile-10.6
// CHECK-CHECK-I386_OSX10_6-NOT: -fobjc-dispatch-method
// CHECK-CHECK-I386_OSX10_6: darwin-objc-defaults

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch i386 -miphoneos-version-min=3.0 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-I386_IPHONE3_0 < %t %s

// CHECK-CHECK-I386_IPHONE3_0: "-cc1"
// CHECK-CHECK-I386_IPHONE3_0: -fobjc-runtime=ios-3.0
// CHECK-CHECK-I386_IPHONE3_0-NOT: -fobjc-dispatch-method
// CHECK-CHECK-I386_IPHONE3_0: darwin-objc-defaults

// x86_64

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch x86_64 -mmacos-version-min=10.4 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-X86_64_OSX10_4 < %t %s

// CHECK-CHECK-X86_64_OSX10_4: "-cc1"
// CHECK-CHECK-X86_64_OSX10_4: -fobjc-dispatch-method=non-legacy

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch x86_64 -mmacos-version-min=10.5 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-X86_64_OSX10_5 < %t %s


// CHECK-CHECK-X86_64_OSX10_5: "-cc1"
// CHECK-CHECK-X86_64_OSX10_5: -fobjc-runtime=macosx-10.5
// CHECK-CHECK-X86_64_OSX10_5: -fobjc-dispatch-method=non-legacy
// CHECK-CHECK-X86_64_OSX10_5: darwin-objc-defaults

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch x86_64 -mmacos-version-min=10.6 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-X86_64_OSX10_6 < %t %s

// CHECK-CHECK-X86_64_OSX10_6: "-cc1"
// CHECK-CHECK-X86_64_OSX10_6: -fobjc-runtime=macosx-10.6
// CHECK-CHECK-X86_64_OSX10_6: darwin-objc-defaults

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch x86_64 -miphoneos-version-min=3.0 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-X86_64_IPHONE3_0 < %t %s

// CHECK-CHECK-X86_64_IPHONE3_0: "-cc1"
// CHECK-CHECK-X86_64_IPHONE3_0: -fobjc-runtime=ios-3.0
// CHECK-CHECK-X86_64_IPHONE3_0: darwin-objc-defaults

// armv7

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch armv7 -mmacos-version-min=10.5 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-ARMV7_OSX10_5 < %t %s

// CHECK-CHECK-ARMV7_OSX10_5: "-cc1"
// CHECK-CHECK-ARMV7_OSX10_5: -fobjc-runtime=macosx-10.5
// CHECK-CHECK-ARMV7_OSX10_5-NOT: -fobjc-dispatch-method
// CHECK-CHECK-ARMV7_OSX10_5: darwin-objc-defaults

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s	\
// RUN:   -arch armv7 -mmacos-version-min=10.6 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-ARMV7_OSX10_6 < %t %s

// CHECK-CHECK-ARMV7_OSX10_6: "-cc1"
// CHECK-CHECK-ARMV7_OSX10_6: -fobjc-runtime=macosx-10.6
// CHECK-CHECK-ARMV7_OSX10_6-NOT: -fobjc-dispatch-method
// CHECK-CHECK-ARMV7_OSX10_6: darwin-objc-defaults

// RUN: %clang -target x86_64-apple-darwin10 -S -### %s \
// RUN:   -arch armv7 -miphoneos-version-min=3.0 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-ARMV7_IPHONE3_0 < %t %s

// CHECK-CHECK-ARMV7_IPHONE3_0: "-cc1"
// CHECK-CHECK-ARMV7_IPHONE3_0: -fobjc-runtime=ios-3.0
// CHECK-CHECK-ARMV7_IPHONE3_0-NOT: -fobjc-dispatch-method
// CHECK-CHECK-ARMV7_IPHONE3_0: darwin-objc-defaults

// RUN: %clang -target x86_64-apple-ios13.1-macabi -S -### %s 2> %t
// RUN: FileCheck --check-prefix CHECK-CHECK-MACCATALYST < %t %s

// CHECK-CHECK-MACCATALYST: "-cc1"
// CHECK-CHECK-MACCATALYST: -fobjc-runtime=macosx-10.15
// CHECK-CHECK-MACCATALYST-NOT: -fobjc-dispatch-method
// CHECK-CHECK-MACCATALYST: darwin-objc-defaults
