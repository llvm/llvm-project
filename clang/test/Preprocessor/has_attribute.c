// RUN: %clang_cc1 -triple arm-unknown-linux -verify -E %s -o - | FileCheck %s

// CHECK: always_inline
#if __has_attribute(always_inline)
int always_inline();
#endif

// CHECK: __always_inline__
#if __has_attribute(__always_inline__)
int __always_inline__();
#endif

// CHECK: no_dummy_attribute
#if !__has_attribute(dummy_attribute)
int no_dummy_attribute();
#endif

// CHECK: has_has_attribute
#ifdef __has_attribute
int has_has_attribute();
#endif

// CHECK: has_something_we_dont_have
#if !__has_attribute(something_we_dont_have)
int has_something_we_dont_have();
#endif

// rdar://10253857
#if __has_attribute(__const)
 int fn3() __attribute__ ((__const));
#endif

#if __has_attribute(const)
 static int constFunction() __attribute__((const));
#endif

// CHECK: has_no_volatile_attribute
#if !__has_attribute(volatile)
int has_no_volatile_attribute();
#endif

// CHECK: has_arm_interrupt
#if __has_attribute(interrupt)
  int has_arm_interrupt();
#endif

// CHECK: does_not_have_dllexport
#if !__has_attribute(dllexport)
  int does_not_have_dllexport();
#endif

// CHECK: does_not_have_uuid
#if !__has_attribute(uuid)
  int does_not_have_uuid
#endif

#if __has_cpp_attribute(selectany) // expected-error {{function-like macro '__has_cpp_attribute' is not defined}}
#endif
