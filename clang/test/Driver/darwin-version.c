// RUN: %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX %s
// CHECK-VERSION-OSX: "armv6k-apple-macosx10.5.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS2 %s
// CHECK-VERSION-IOS2: "armv6k-apple-ios2.0.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=2.2 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS22 %s
// CHECK-VERSION-IOS22: "armv6k-apple-ios2.2.0"
// RUN: %clang -target armv6-apple-darwin9 -miphoneos-version-min=3.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS3 %s
// CHECK-VERSION-IOS3: "armv6k-apple-ios3.0.0"
// RUN: %clang -target i686-apple-darwin8 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.4 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX4 %s
// CHECK-VERSION-OSX4: "i386-apple-macosx10.4.0"
// RUN: %clang -target i686-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.5 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// CHECK-VERSION-OSX5: "i386-apple-macosx10.5.0"
// RUN: %clang -target i686-apple-darwin10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
// RUN: %clang -target i686-apple-darwin9 -mmacosx-version-min=10.6 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX6 %s
// CHECK-VERSION-OSX6: "i386-apple-macosx10.6.0"
// RUN: %clang -target x86_64-apple-darwin14 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// RUN: %clang -target x86_64-apple-macosx -mmacosx-version-min=10.10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// RUN: %clang -target x86_64-apple-macosx -mmacos-version-min=10.10 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX10 %s
// CHECK-VERSION-OSX10: "x86_64-apple-macosx10.10.0"
// RUN: %clang -target x86_64-apple-macosx -mmacosx-version-min= -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-MISSING %s
// RUN: %clang -target x86_64-apple-macosx -mmacos-version-min= -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-MISSING %s
// CHECK-VERSION-MISSING: invalid version number
// RUN: %clang -target armv7k-apple-darwin -mwatchos-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOS20 %s
// RUN: %clang -target armv7-apple-darwin -mtvos-version-min=8.3 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVOS83 %s
// CHECK-VERSION-TVOS83: "thumbv7-apple-tvos8.3.0"
// RUN: %clang -target i386-apple-darwin -mtvos-simulator-version-min=8.3 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVSIM83 %s
// CHECK-VERSION-TVSIM83: "i386-apple-tvos8.3.0"
// RUN: %clang -target armv7k-apple-darwin -mwatchos-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOS20 %s
// CHECK-VERSION-WATCHOS20: "thumbv7k-apple-watchos2.0.0"
// RUN: %clang -target i386-apple-darwin -mwatchos-simulator-version-min=2.0 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHSIM20 %s
// CHECK-VERSION-WATCHSIM20: "i386-apple-watchos2.0.0"

// Check environment variable gets interpreted correctly
// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 IPHONEOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX5 %s
// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 IPHONEOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS2 %s

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.4.10 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-OSX49 %s
// CHECK-VERSION-OSX49: "i386-apple-macosx10.4.10"
// RUN: env IPHONEOS_DEPLOYMENT_TARGET=2.3.1 \
// RUN:   %clang -target armv6-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-IOS231 %s
// CHECK-VERSION-IOS231: "armv6k-apple-ios2.3.1"

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 TVOS_DEPLOYMENT_TARGET=8.3.1 \
// RUN:   %clang -target armv7-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVOS %s
// CHECK-VERSION-TVOS: "thumbv7-apple-tvos8.3.1"
// RUN: env TVOS_DEPLOYMENT_TARGET=8.3.1 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-TVOSSIM %s
// CHECK-VERSION-TVOSSIM: "i386-apple-tvos8.3.1"

// RUN: env MACOSX_DEPLOYMENT_TARGET=10.5 WATCHOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target armv7-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOS %s
// CHECK-VERSION-WATCHOS: "thumbv7-apple-watchos2.0.0"
// RUN: env WATCHOS_DEPLOYMENT_TARGET=2.0 \
// RUN:   %clang -target i386-apple-darwin9 -c %s -### 2>&1 | \
// RUN:   FileCheck --check-prefix=CHECK-VERSION-WATCHOSSIM %s
// CHECK-VERSION-WATCHOSSIM: "i386-apple-watchos2.0.0"
