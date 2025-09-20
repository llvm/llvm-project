// RUN: %clang_cc1 %s -Winteger-overflow -Wno-unused-value -foverflow-behavior-types -Woverflow-behavior-conversion -Wconstant-conversion -verify -fsyntax-only -std=c11 -Wno-pointer-sign

typedef int __attribute__((overflow_behavior)) bad_arg_count; // expected-error {{'overflow_behavior' attribute takes one argument}}
typedef int __attribute__((overflow_behavior(not_real))) bad_arg_spec; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef int __attribute__((overflow_behavior("not_real"))) bad_arg_spec_str; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef char* __attribute__((overflow_behavior("wrap"))) bad_type; // expected-warning {{'overflow_behavior' attribute cannot be applied to non-integer type 'char *'; attribute ignored}}

typedef int __attribute__((overflow_behavior(wrap))) ok_wrap; // OK
typedef long __attribute__((overflow_behavior(no_wrap))) ok_nowrap; // OK
typedef unsigned long __attribute__((overflow_behavior("wrap"))) str_ok_wrap; // OK
typedef char __attribute__((overflow_behavior("no_wrap"))) str_ok_nowrap; // OK

void foo() {
  (2147483647 + 100); // expected-warning {{overflow in expression; result is }}
  (ok_wrap)2147483647 + 100; // no warn
}

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_wrap __attribute__((overflow_behavior(no_wrap)))

void ptr(int a) {
  int __no_wrap *p = &a; // expected-warning {{initializing '__no_wrap int *' with an expression of type 'int *' discards overflow behavior}}
}

void ptr2(__no_wrap int a) {
  int *p = &a; // expected-warning {{initializing 'int *' with an expression of type '__no_wrap int *' discards overflow behavior}}
}


// verify semantics of -Wimplicitly-discarded-overflow-behavior{,-pedantic}
void imp_disc_pedantic(unsigned a) {}
void imp_disc(int a) {}
void imp_disc_test(unsigned __attribute__((overflow_behavior(wrap))) a) {
  imp_disc_pedantic(a); // expected-warning {{implicit conversion from '__wrap unsigned int' to 'unsigned int' discards overflow behavior}}
  imp_disc(a); // expected-warning {{implicit conversion from '__wrap unsigned int' to 'int' discards overflow behavior}}
}

// -Wconversion for assignments that discard overflow behavior
void assignment_disc(unsigned __attribute__((overflow_behavior(wrap))) a) {
  int b = a; // expected-warning {{implicit conversion from '__wrap unsigned int' to 'int' during assignment discards overflow behavior}}
  int c = (int)a; // OK
}

void constant_conversion() {
  // expected-warning@+2 {{implicit conversion from '__wrap int' to 'short' changes value from 100000 to -31072}}
  // expected-warning@+1 {{implicit conversion from '__wrap int' to 'short' during assignment discards overflow behavior}}
  short x1 = (int __wrap)100000;
  short __wrap x2 = (int)100000; // No warning expected
  // expected-warning@+1 {{implicit conversion from 'int' to '__no_wrap short' changes value from 100000 to -31072}}
  short __no_wrap x3 = (int)100000;
  // expected-warning@+2 {{implicit conversion from '__no_wrap int' to 'short' changes value from 100000 to -31072}}
  // expected-warning@+1 {{implicit conversion from '__no_wrap int' to 'short' during assignment discards overflow behavior}}
  short x4 = (int __no_wrap)100000;

  unsigned short __wrap ux1 = (unsigned int)100000; // No warning - wrapping expected
  // expected-warning@+2 {{implicit conversion from '__wrap unsigned int' to 'unsigned short' changes value from 100000 to 34464}}
  // expected-warning@+1 {{implicit conversion from '__wrap unsigned int' to 'unsigned short' discards overflow behavior}}
  unsigned short ux2 = (unsigned int __wrap)100000;
  unsigned short __no_wrap ux3 = (unsigned int)100000; // expected-warning {{implicit conversion from 'unsigned int' to '__no_wrap unsigned short' changes value from 100000 to 34464}}
  unsigned short __no_wrap ux4 = (unsigned int __no_wrap)100000; // expected-warning {{implicit conversion from '__no_wrap unsigned int' to '__no_wrap unsigned short' changes value from 100000 to 34464}}
  unsigned short __no_wrap ux5 = (unsigned int __wrap)100000; // expected-warning {{implicit conversion from '__wrap unsigned int' to '__no_wrap unsigned short' changes value from 100000 to 34464}}
}

typedef long s64_typedef1;
typedef s64_typedef1 __attribute__((overflow_behavior(no_wrap))) nw_s64_typedef2;
nw_s64_typedef2 global_var;
void test_nested_typedef_control_flow() {
  // We had a crash during Sema with nested typedefs and control flow, make
  // sure we don't crash and just warn.
  if (global_var) {} // expected-warning {{implicit conversion from 'nw_s64_typedef2'}}
}

int test_discard_on_return(unsigned long __no_wrap a) {
  return a; // expected-warning {{implicit conversion from '__no_wrap unsigned long' to 'int' discards overflow behavior}}
}

// Test OBT pointer compatibility
void test_obt_pointer_compatibility() {
  unsigned long x = 42;
  unsigned long __no_wrap y = 42;
  unsigned long __wrap z = 42;

  unsigned long *px = &x;
  unsigned long __no_wrap *py = &y;
  unsigned long __wrap *pz = &z;

  // Same types - should not warn
  px = &x; // OK
  py = &y; // OK
  pz = &z; // OK

  // Different OBT annotations - should warn but allow
  // expected-warning@+1 {{assigning to 'unsigned long *' from '__no_wrap unsigned long *' discards overflow behavior}}
  px = py;
  // expected-warning@+1 {{assigning to 'unsigned long *' from '__wrap unsigned long *' discards overflow behavior}}
  px = pz;
  // expected-warning@+1 {{assigning to '__no_wrap unsigned long *' from 'unsigned long *' discards overflow behavior}}
  py = px;
  // expected-warning@+1 {{assigning to '__wrap unsigned long *' from 'unsigned long *' discards overflow behavior}}
  pz = px;
  // expected-warning@+1 {{assigning to '__no_wrap unsigned long *' from '__wrap unsigned long *' discards overflow behavior}}
  py = pz;
  // expected-warning@+1 {{assigning to '__wrap unsigned long *' from '__no_wrap unsigned long *' discards overflow behavior}}
  pz = py;
}

// Test function parameter passing
// expected-note@+2 {{passing argument to parameter 'p' here}}
// expected-note@+1 {{passing argument to parameter 'p' here}}
void func_takes_regular_ptr(unsigned long *p) {}
// expected-note@+1 {{passing argument to parameter 'p' here}}
void func_takes_no_wrap_ptr(unsigned long __no_wrap *p) {}
  // expected-note@+1 {{passing argument to parameter 'p' here}}
void func_takes_wrap_ptr(unsigned long __wrap *p) {}

void test_function_parameters() {
  unsigned long x = 42;
  unsigned long __no_wrap y = 42;
  unsigned long __wrap z = 42;

  unsigned long *px = &x;
  unsigned long __no_wrap *py = &y;
  unsigned long __wrap *pz = &z;

  // Same types - should not warn
  func_takes_regular_ptr(px); // OK
  func_takes_no_wrap_ptr(py); // OK
  func_takes_wrap_ptr(pz); // OK

  // Different OBT annotations - should warn but allow
  // expected-warning@+1 {{passing '__no_wrap unsigned long *' to parameter of type 'unsigned long *' discards overflow behavior}}
  func_takes_regular_ptr(py);
  // expected-warning@+1 {{passing '__wrap unsigned long *' to parameter of type 'unsigned long *' discards overflow behavior}}
  func_takes_regular_ptr(pz);
  // expected-warning@+1 {{passing 'unsigned long *' to parameter of type '__no_wrap unsigned long *' discards overflow behavior}}
  func_takes_no_wrap_ptr(px);
  // expected-warning@+1 {{passing 'unsigned long *' to parameter of type '__wrap unsigned long *' discards overflow behavior}}
  func_takes_wrap_ptr(px);
}

void test_different_underlying_types_for_pointers() {
  int x = 42;
  unsigned long __no_wrap y = 42;

  int *px = &x;
  unsigned long __no_wrap *py = &y;

  px = py; // expected-error {{incompatible pointer types assigning to 'int *' from '__no_wrap unsigned long *'}}
}

typedef unsigned long __no_wrap nw_ul;
typedef signed long sl;

void qux(nw_ul *ptr) {}

void test_signed_unsigned_pointer_compatibility() {
  sl a;
  qux(&a);
}

// expected-warning@+1 {{conflicting 'overflow_behavior' attributes on the same type; 'no_wrap' takes precedence over 'wrap'}}
typedef int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(no_wrap))) conflicting_no_wrap_wins;
// expected-warning@+1 {{conflicting 'overflow_behavior' attributes on the same type; 'no_wrap' takes precedence over 'wrap'}}
typedef int __attribute__((overflow_behavior(no_wrap))) __attribute__((overflow_behavior(wrap))) conflicting_wrap_ignored;

void test_conflicting_behavior_kinds() {
  conflicting_no_wrap_wins x = 42;
  conflicting_wrap_ignored y = 42;

  int __no_wrap *px = &x;
  int __no_wrap *py = &y;
}

typedef int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(wrap))) duplicate_wrap; // no warn
typedef int __attribute__((overflow_behavior(no_wrap))) __attribute__((overflow_behavior(no_wrap))) duplicate_no_wrap; // no warn

// Test type merging behavior for OBTs on top of typedefs
typedef int pid_t;
typedef int clockid_t;

void test_obt_type_merging() {
  pid_t __wrap a = 1;
  clockid_t __wrap b = 2;
  pid_t __wrap c = 4;
  _Static_assert(_Generic((a + b), int __wrap: 1, default: 0), "a + b should be __wrap int");
  _Static_assert(_Generic((a + c), pid_t __wrap: 1, default: 0), "a + c should be __wrap pid_t");
}

typedef unsigned long __no_wrap test_size_t;
typedef int __wrap test_wrap_int;

void test_pointer_arithmetic_crash_fix() {
  int a = 42;
  test_size_t offset = 10;
  test_wrap_int w_offset = 5;

  int *ptr1 = &a + offset;
  int *ptr2 = &a + w_offset;
  int *ptr3 = offset + &a;

  test_size_t bad1 = &a + offset;     // expected-error {{incompatible pointer to integer conversion}}
  test_wrap_int bad2 = &a + w_offset; // expected-error {{incompatible pointer to integer conversion}}
  test_size_t b = &a + b;       // expected-error {{incompatible pointer to integer conversion}}
  int arr[10];
  test_size_t diff = &arr[5] - &arr[0]; // OK: pointer difference assigned to OBT
}
