// REQUIRES: x86-registered-target
// REQUIRES: arm-registered-target
// REQUIRES: aarch64-registered-target

// RUN: %clang -target x86_64-apple-darwin -Wincompatible-sysroot -isysroot SDKs/MacOSX10.9.sdk -mios-version-min=9.0 -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-OSX-IOS %s
// RUN: %clang -target arm64-apple-darwin -Wincompatible-sysroot -isysroot SDKs/iPhoneOS9.2.sdk -mwatchos-version-min=2.0 -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-IOS-WATCHOS %s
// RUN: %clang -target arm64-apple-darwin -Wincompatible-sysroot -isysroot SDKs/iPhoneOS9.2.sdk -mtvos-version-min=9.0 -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-IOS-TVOS %s
// RUN: %clang -target x86_64-apple-driverkit19.0 -Wincompatible-sysroot -isysroot SDKs/MacOSX10.9.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-OSX-DRIVERKIT %s
// RUN: %clang -target x86_64-apple-driverkit19.0 -Wincompatible-sysroot -isysroot SDKs/iPhoneOS9.2.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-IOS-DRIVERKIT %s
// RUN: %clang -target x86_64-apple-darwin -Wincompatible-sysroot -isysroot SDKs/iPhoneSimulator9.2.sdk -mios-version-min=9.0 -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-IOS-IOSSIM %s
// RUN: %clang -target x86_64-apple-darwin -Wno-incompatible-sysroot -isysroot SDKs/MacOSX10.9.sdk -mios-version-min=9.0 -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-OSX-IOS-DISABLED %s

// RUN: %clang -target armv7k-apple-watchos6.0 -Wincompatible-sysroot -isysroot %S/Inputs/WatchOS6.0.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-WATCHOS %s
// RUN: %clang -target armv7-apple-watchos6.0 -Wincompatible-sysroot -isysroot %S/Inputs/WatchOS6.0.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-WATCHOS-ARMV7 %s
// RUN: %clang -target armv6m-apple-firmware -Wincompatible-sysroot -isysroot %S/Inputs/WatchOS6.0.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-WATCHOS-ARMV6M-FIRMWARE %s
// RUN: %clang -target arm64-apple-visionos1.0-simulator -Wincompatible-sysroot -isysroot %S/Inputs/XRSimulator1.0.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-VISIONOSSIM %s
// RUN: %clang -target arm64-apple-xros1.0 -Wincompatible-sysroot -isysroot %S/Inputs/XRSimulator1.0.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-VISIONOSSIM-VISIONOS %s
// RUN: %clang -target arm64-apple-ios17.1 -Wincompatible-sysroot -isysroot %S/Inputs/XRSimulator1.0.sdk -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-VISIONOSSIM-IOS %s
// RUN: %clang -target arm64-apple-visionos1.0-simulator -Wincompatible-sysroot -isysroot %S/Inputs/XRSimulator1.0.sdk/usr/include/libxml -### -o - %s 2>&1 | FileCheck -check-prefix CHECK-VISIONOSSIM %s

int main() { return 0; }
// CHECK-OSX-IOS: warning: using sysroot for 'MacOSX10.9' but targeting 'x86_64-apple-ios9.0.0-simulator'
// CHECK-IOS-WATCHOS: warning: using sysroot for 'iPhoneOS9.2' but targeting 'arm64-apple-watchos2.0.0'
// CHECK-IOS-TVOS: warning: using sysroot for 'iPhoneOS9.2' but targeting 'arm64-apple-tvos9.0.0'
// CHECK-OSX-DRIVERKIT: warning: using sysroot for 'MacOSX10.9' but targeting 'x86_64-apple-driverkit19.0.0'
// CHECK-IOS-DRIVERKIT: warning: using sysroot for 'iPhoneOS9.2' but targeting 'x86_64-apple-driverkit19.0.0'
// CHECK-IOS-IOSSIM-NOT: warning: using sysroot for '{{.*}}' but targeting '{{.*}}'
// CHECK-OSX-IOS-DISABLED-NOT: warning: using sysroot for '{{.*}}' but targeting '{{.*}}'

// CHECK-WATCHOS-NOT: warning: using sysroot for '{{.*}}' but targeting '{{.*}}'
// CHECK-WATCHOS-ARMV7: warning: using sysroot for 'watchOS 6.0' but targeting 'thumbv7-apple-watchos6.0.0'
// CHECK-WATCHOS-ARMV6M-FIRMWARE: warning: using sysroot for 'watchOS 6.0' but targeting 'thumbv6m-apple-firmware1.0.0'
// CHECK-VISIONOSSIM-NOT: warning: using sysroot for '{{.*}}' but targeting '{{.*}}'
// CHECK-VISIONOSSIM-VISIONOS: warning: using sysroot for 'Simulator - visionOS 1.0' but targeting 'arm64-apple-xros1.0.0'
// CHECK-VISIONOSSIM-IOS: warning: using sysroot for 'Simulator - visionOS 1.0' but targeting 'arm64-apple-ios17.1.0'
