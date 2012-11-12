// RUN: %clang_cc1 -x objective-c -fobjc-exceptions -fexceptions -E -dM %s | FileCheck -check-prefix=CHECK-OBJC-NOCXX %s 
// CHECK-OBJC-NOCXX: #define OBJC_ZEROCOST_EXCEPTIONS 1
// CHECK-OBJC-NOCXX-NOT: #define __EXCEPTIONS 1

// RUN: %clang_cc1 -x objective-c++ -fobjc-exceptions -fexceptions -fcxx-exceptions -E -dM %s | FileCheck -check-prefix=CHECK-OBJC-CXX %s 
// CHECK-OBJC-CXX: #define OBJC_ZEROCOST_EXCEPTIONS 1
// CHECK-OBJC-CXX: #define __EXCEPTIONS 1

// RUN: %clang_cc1 -x objective-c++ -fexceptions -fcxx-exceptions -E -dM %s | FileCheck -check-prefix=CHECK-NOOBJC-CXX %s 
// CHECK-NOOBJC-CXX-NOT: #define OBJC_ZEROCOST_EXCEPTIONS 1
// CHECK-NOOBJC-CXX: #define __EXCEPTIONS 1

// RUN: %clang_cc1 -x objective-c -E -dM %s | FileCheck -check-prefix=CHECK-NOOBJC-NOCXX %s 
// CHECK-NOOBJC-NOCXX-NOT: #define OBJC_ZEROCOST_EXCEPTIONS 1
// CHECK-NOOBJC-NOCXX-NOT: #define __EXCEPTIONS 1
