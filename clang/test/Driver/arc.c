// RUN: %clang -ObjC -target i386-apple-darwin10 -m32 -fobjc-arc %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -x objective-c -target i386-apple-darwin10 -m32 -fobjc-arc %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -x objective-c++ -target i386-apple-darwin10 -m32 -fobjc-arc %s -fsyntax-only 2>&1 | FileCheck %s
// RUN: %clang -x c -target i386-apple-darwin10 -m32 -fobjc-arc %s -fsyntax-only 2>&1 | FileCheck -check-prefix NOTOBJC %s
// RUN: %clang -x c++ -target i386-apple-darwin10 -m32 -fobjc-arc %s -fsyntax-only 2>&1 | FileCheck -check-prefix NOTOBJC %s
// RUN: %clang -x objective-c -target x86_64-apple-darwin11 -mmacosx-version-min=10.5 -fobjc-arc %s -fsyntax-only 2>&1 | FileCheck -check-prefix UNSUPPORTED %s

// Just to test clang is working.
# foo

// CHECK: error: -fobjc-arc is not supported on platforms using the legacy runtime
// CHECK-NOT: invalid preprocessing directive

// NOTOBJC-NOT: error: -fobjc-arc is not supported on platforms using the legacy runtime
// NOTOBJC: invalid preprocessing directive

// UNSUPPORTED: error: -fobjc-arc is not supported on versions of OS X prior to 10.6
