// RUN: %clang_cc1 -E %s -o - | FileCheck %s

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

