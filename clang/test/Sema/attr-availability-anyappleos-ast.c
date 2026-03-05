// RUN: %clang_cc1 -triple arm64-apple-ios26.0 -fsyntax-only -ast-dump %s | FileCheck --check-prefix=CHECK-IOS %s
// RUN: %clang_cc1 -triple x86_64-apple-macosx26.0 -fsyntax-only -ast-dump %s | FileCheck --check-prefix=CHECK-MACOS %s
// RUN: %clang_cc1 -triple arm64-apple-tvos26.0 -fsyntax-only -ast-dump %s | FileCheck --check-prefix=CHECK-TVOS %s
// RUN: %clang_cc1 -triple arm64-apple-watchos26.0 -fsyntax-only -ast-dump %s | FileCheck --check-prefix=CHECK-WATCHOS %s
// RUN: %clang_cc1 -triple arm64-apple-xros26.0 -fsyntax-only -ast-dump %s | FileCheck --check-prefix=CHECK-XROS %s
// RUN: %clang_cc1 -triple arm64-apple-ios26.0-macabi -fsyntax-only -ast-dump %s | FileCheck --check-prefix=CHECK-MACCATALYST %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only -ast-dump %s | FileCheck --check-prefix=CHECK-LINUX %s

// Test that anyAppleOS availability creates implicit platform-specific
// attributes for the target platform.

extern int func1 __attribute__((availability(anyAppleOS, introduced=26.0)));
// CHECK-IOS: AvailabilityAttr {{.*}} Implicit ios 26.0 0 0 "" "" 3
// CHECK-MACOS: AvailabilityAttr {{.*}} Implicit macos 26.0 0 0 "" "" 3
// CHECK-TVOS: AvailabilityAttr {{.*}} Implicit tvos 26.0 0 0 "" "" 3
// CHECK-WATCHOS: AvailabilityAttr {{.*}} Implicit watchos 26.0 0 0 "" "" 3
// CHECK-XROS: AvailabilityAttr {{.*}} Implicit xros 26.0 0 0 "" "" 3
// CHECK-MACCATALYST: AvailabilityAttr {{.*}} Implicit maccatalyst 26.0 0 0 "" "" 3
// CHECK-LINUX-NOT: AvailabilityAttr {{.*}} Implicit

extern int func2 __attribute__((availability(anyAppleOS, introduced=26.0, deprecated=27.0)));
// CHECK-IOS: AvailabilityAttr {{.*}} Implicit ios 26.0 27.0 0 "" "" 3
// CHECK-MACOS: AvailabilityAttr {{.*}} Implicit macos 26.0 27.0 0 "" "" 3

extern int func3 __attribute__((availability(anyAppleOS, introduced=26.0, deprecated=27.0, obsoleted=28.0)));
// CHECK-IOS: AvailabilityAttr {{.*}} Implicit ios 26.0 27.0 28.0 "" "" 3
// CHECK-MACOS: AvailabilityAttr {{.*}} Implicit macos 26.0 27.0 28.0 "" "" 3

extern int func4 __attribute__((availability(anyAppleOS, unavailable)));
// CHECK-IOS: AvailabilityAttr {{.*}} Implicit ios 0 0 0 Unavailable "" "" 3
// CHECK-MACOS: AvailabilityAttr {{.*}} Implicit macos 0 0 0 Unavailable "" "" 3

extern int func5 __attribute__((availability(anyAppleOS, unavailable, message="Use something else")));
// CHECK-IOS: AvailabilityAttr {{.*}} Implicit ios 0 0 0 Unavailable "Use something else" "" 3
// CHECK-MACOS: AvailabilityAttr {{.*}} Implicit macos 0 0 0 Unavailable "Use something else" "" 3
