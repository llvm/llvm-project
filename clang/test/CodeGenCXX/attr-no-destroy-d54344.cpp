// RUN: %clang_cc1 -std=c++2a -emit-llvm -O0 -triple x86_64-unknown-linux-gnu -DNOATTR  %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++2a -emit-llvm -O0 -triple x86_64-unknown-linux-gnu %s -o - | FileCheck %s --check-prefix=CHECK-ATTR
// RUN: %clang_cc1 -std=c++2a -emit-llvm -O0 -triple x86_64-unknown-linux-gnu -DNOATTR -fno-c++-static-destructors %s -o - | FileCheck %s --check-prefix=CHECK-FLAG

// Regression test for D54344. Class with no user-defined destructor
// that has an inherited member that has a non-trivial destructor
// and a non-default constructor will attempt to emit a destructor
// despite being marked as __attribute((no_destroy)) in which case
// it would trigger an assertion due to an incorrect assumption.

// This test is more reliable with asserts to work as without 
// the crash may (unlikely) could generate working but semantically
// incorrect code.

class a {
public:
  a();
  ~a();
};
class logger_base {
  a d;
};
class e : logger_base {};
#ifndef NOATTR
__attribute((no_destroy))
#endif
e g;

// In the absence of the attribute and flag, both ctor and dtor should
// be emitted, check for that.
// CHECK: @__cxx_global_var_init
// CHECK: @__cxa_atexit

// When attribute is enabled, the constructor should not be balanced
// by a destructor. Make sure we have the ctor but not the dtor
// registration.
// CHECK-ATTR: @__cxx_global_var_init
// CHECK-ATTR-NOT: @__cxa_atexit

// Same scenario except with global flag (-fno-c++-static-destructors)
// supressing it instead of the attribute. 
// CHECK-FLAG: @__cxx_global_var_init
// CHECK-FLAG-NOT: @__cxa_atexit
