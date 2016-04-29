// RUN: %clang -Wincompatible-sysroot -isysroot SDKs/MacOSX10.9.sdk -mios-version-min=9.0 -S -o - %s 2>&1 | FileCheck -check-prefix CHECK-OSX-IOS %s
// RUN: %clang -Wincompatible-sysroot -isysroot SDKs/iPhoneOS9.2.sdk -mwatchos-version-min=2.0 -S -o - %s 2>&1 | FileCheck -check-prefix CHECK-IOS-WATCHOS %s
// RUN: %clang -Wincompatible-sysroot -isysroot SDKs/iPhoneOS9.2.sdk -mtvos-version-min=9.0 -S -o - %s 2>&1 | FileCheck -check-prefix CHECK-IOS-TVOS %s
// RUN: %clang -Wincompatible-sysroot -isysroot SDKs/iPhoneSimulator9.2.sdk -mios-version-min=9.0 -S -o - %s 2>&1 | FileCheck -check-prefix CHECK-IOS-IOSSIM %s
// RUN: %clang -Wno-incompatible-sysroot -isysroot SDKs/MacOSX10.9.sdk -mios-version-min=9.0 -S -o - %s 2>&1 | FileCheck -check-prefix CHECK-OSX-IOS-DISABLED %s

int main() { return 0; }
// CHECK-OSX-IOS: warning: using sysroot for 'MacOSX' but targeting 'iPhone'
// CHECK-IOS-WATCHOS: warning: using sysroot for 'iPhoneOS' but targeting 'Watch'
// CHECK-IOS-TVOS: warning: using sysroot for 'iPhoneOS' but targeting 'AppleTV'
// CHECK-IOS-IOSSIM-NOT: warning: using sysroot for '{{.*}}' but targeting '{{.*}}'
// CHECK-OSX-IOS-DISABLED-NOT: warning: using sysroot for '{{.*}}' but targeting '{{.*}}'
