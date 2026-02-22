// RUN: %clang_cc1 %s -Winteger-overflow -Wno-unused-value -fexperimental-overflow-behavior-types -Woverflow-behavior-conversion -Wconstant-conversion -verify -fsyntax-only -std=c11 -Wno-pointer-sign

typedef int __attribute__((overflow_behavior)) bad_arg_count; // expected-error {{'overflow_behavior' attribute takes one argument}}
typedef int __attribute__((overflow_behavior(not_real))) bad_arg_spec; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef int __attribute__((overflow_behavior("not_real"))) bad_arg_spec_str; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef char* __attribute__((overflow_behavior("wrap"))) bad_type; // expected-error {{'overflow_behavior' attribute cannot be applied to non-integer type 'char *'}}

typedef int __attribute__((overflow_behavior(wrap))) ok_wrap; // OK
typedef long __attribute__((overflow_behavior(trap))) ok_nowrap; // OK
typedef unsigned long __attribute__((overflow_behavior("wrap"))) str_ok_wrap; // OK
typedef char __attribute__((overflow_behavior("trap"))) str_ok_nowrap; // OK

void foo() {
  (2147483647 + 100); // expected-warning {{overflow in expression; result is }}
  (ok_wrap)2147483647 + 100; // no warn
}

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __trap __attribute__((overflow_behavior(trap)))

void ptr(int a) {
  int __ob_trap *p = &a; // expected-warning {{initializing '__ob_trap int *' with an expression of type 'int *' discards overflow behavior}}
}

void ptr2(__ob_trap int a) {
  int *p = &a; // expected-warning {{initializing 'int *' with an expression of type '__ob_trap int *' discards overflow behavior}}
}


// verify semantics of OBT at function boundaries
void imp_disc_pedantic(unsigned a) {}
void imp_disc(int a) {}
void imp_disc_test(unsigned __attribute__((overflow_behavior(wrap))) a) {
  imp_disc_pedantic(a); // expected-warning {{passing argument of type '__ob_wrap unsigned int' to parameter of type 'unsigned int' discards overflow behavior at function boundary}}
  // expected-warning@-1 {{implicit conversion from '__ob_wrap unsigned int' to 'unsigned int' discards overflow behavior}}
  imp_disc(a); // expected-warning {{passing argument of type '__ob_wrap unsigned int' to parameter of type 'int' discards overflow behavior at function boundary}}
  // expected-warning@-1 {{implicit conversion from '__ob_wrap unsigned int' to 'int' discards overflow behavior}}
}

void assignment_disc(unsigned __attribute__((overflow_behavior(wrap))) a) {
  int b = a; // expected-warning {{implicit conversion from '__ob_wrap unsigned int' to 'int' during assignment discards overflow behavior}}
  int c = (int)a; // OK
}

void constant_conversion() {
  short x1 = (int __ob_wrap)100000;
  short __ob_wrap x2 = (int)100000; // No warning expected
  // expected-warning@+1 {{implicit conversion from 'int' to '__ob_trap short' changes value from 100000 to -31072}}
  short __ob_trap x3 = (int)100000;
  // OBT now propagates through implicit casts, so x4 gets __ob_trap from the explicit cast
  // expected-warning@+1 {{implicit conversion from '__ob_trap int' to '__ob_trap short' changes value from 100000 to -31072}}
  short x4 = (int __ob_trap)100000;

  unsigned short __ob_wrap ux1 = (unsigned int)100000; // No warning - wrapping expected
  unsigned short ux2 = (unsigned int __ob_wrap)100000;
  unsigned short __ob_trap ux3 = (unsigned int)100000; // expected-warning {{implicit conversion from 'unsigned int' to '__ob_trap unsigned short' changes value from 100000 to 34464}}
  unsigned short __ob_trap ux4 = (unsigned int __ob_trap)100000; // expected-warning {{implicit conversion from '__ob_trap unsigned int' to '__ob_trap unsigned short' changes value from 100000 to 34464}}
  unsigned short __ob_trap ux5 = (unsigned int __ob_wrap)100000; // expected-error {{assigning to '__ob_trap unsigned short' from '__ob_wrap unsigned int' with incompatible overflow behavior types ('__ob_trap' and '__ob_wrap')}}
}

typedef long s64_typedef1;
typedef s64_typedef1 __attribute__((overflow_behavior(trap))) nw_s64_typedef2;
nw_s64_typedef2 global_var;
void test_nested_typedef_control_flow() {
  // We had a crash during Sema with nested typedefs and control flow, make
  // sure we don't crash and just warn.
  if (global_var) {} // expected-warning {{implicit conversion from 'nw_s64_typedef2'}}
}

int test_discard_on_return(unsigned long __ob_trap a) {
  return a; // expected-warning {{implicit conversion from '__ob_trap unsigned long' to 'int' discards overflow behavior}}
}

// Test OBT pointer compatibility
void test_obt_pointer_compatibility() {
  unsigned long x = 42;
  unsigned long __ob_trap y = 42;
  unsigned long __ob_wrap z = 42;

  unsigned long *px = &x;
  unsigned long __ob_trap *py = &y;
  unsigned long __ob_wrap *pz = &z;

  // Same types - should not warn
  px = &x; // OK
  py = &y; // OK
  pz = &z; // OK

  // Different OBT annotations - should warn but allow
  // expected-warning@+1 {{assigning to 'unsigned long *' from '__ob_trap unsigned long *' discards overflow behavior}}
  px = py;
  // expected-warning@+1 {{assigning to 'unsigned long *' from '__ob_wrap unsigned long *' discards overflow behavior}}
  px = pz;
  // expected-warning@+1 {{assigning to '__ob_trap unsigned long *' from 'unsigned long *' discards overflow behavior}}
  py = px;
  // expected-warning@+1 {{assigning to '__ob_wrap unsigned long *' from 'unsigned long *' discards overflow behavior}}
  pz = px;
  // expected-error@+1 {{assigning to '__ob_trap unsigned long *' from '__ob_wrap unsigned long *' with incompatible overflow behavior types ('__ob_trap' and '__ob_wrap')}}
  py = pz;
  // expected-error@+1 {{assigning to '__ob_wrap unsigned long *' from '__ob_trap unsigned long *' with incompatible overflow behavior types ('__ob_wrap' and '__ob_trap')}}
  pz = py;
}

// Test function parameter passing
// expected-note@+2 {{passing argument to parameter 'p' here}}
// expected-note@+1 {{passing argument to parameter 'p' here}}
void func_takes_regular_ptr(unsigned long *p) {}
// expected-note@+1 {{passing argument to parameter 'p' here}}
void func_takes_trap_ptr(unsigned long __ob_trap *p) {}
  // expected-note@+1 {{passing argument to parameter 'p' here}}
void func_takes_wrap_ptr(unsigned long __ob_wrap *p) {}

void test_function_parameters() {
  unsigned long x = 42;
  unsigned long __ob_trap y = 42;
  unsigned long __ob_wrap z = 42;

  unsigned long *px = &x;
  unsigned long __ob_trap *py = &y;
  unsigned long __ob_wrap *pz = &z;

  // Same types - should not warn
  func_takes_regular_ptr(px); // OK
  func_takes_trap_ptr(py); // OK
  func_takes_wrap_ptr(pz); // OK

  // Different OBT annotations - should warn but allow
  // expected-warning@+1 {{passing '__ob_trap unsigned long *' to parameter of type 'unsigned long *' discards overflow behavior}}
  func_takes_regular_ptr(py);
  // expected-warning@+1 {{passing '__ob_wrap unsigned long *' to parameter of type 'unsigned long *' discards overflow behavior}}
  func_takes_regular_ptr(pz);
  // expected-warning@+1 {{passing 'unsigned long *' to parameter of type '__ob_trap unsigned long *' discards overflow behavior}}
  func_takes_trap_ptr(px);
  // expected-warning@+1 {{passing 'unsigned long *' to parameter of type '__ob_wrap unsigned long *' discards overflow behavior}}
  func_takes_wrap_ptr(px);
}

void test_different_underlying_types_for_pointers() {
  int x = 42;
  unsigned long __ob_trap y = 42;

  int *px = &x;
  unsigned long __ob_trap *py = &y;

  px = py; // expected-error {{incompatible pointer types assigning to 'int *' from '__ob_trap unsigned long *'}}
}

typedef unsigned long __ob_trap nw_ul;
typedef signed long sl;

void qux(nw_ul *ptr) {}

void test_signed_unsigned_pointer_compatibility() {
  sl a;
  qux(&a);
}

// expected-error@+1 {{conflicting 'overflow_behavior' attributes on the same type}}
typedef int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(trap))) conflicting_trap_wins;
// expected-error@+1 {{conflicting 'overflow_behavior' attributes on the same type}}
typedef int __attribute__((overflow_behavior(trap))) __attribute__((overflow_behavior(wrap))) conflicting_wrap_ignored;

void test_conflicting_behavior_kinds() {
  conflicting_trap_wins x = 42;
  conflicting_wrap_ignored y = 42;

  int __ob_trap *px = &x;
  int __ob_trap *py = &y;
}

typedef int __attribute__((overflow_behavior(wrap))) __attribute__((overflow_behavior(wrap))) duplicate_wrap; // no warn
typedef int __attribute__((overflow_behavior(trap))) __attribute__((overflow_behavior(trap))) duplicate_trap; // no warn

// Test type merging behavior for OBTs on top of typedefs
typedef int pid_t;
typedef int clockid_t;

void test_obt_type_merging() {
  pid_t __ob_wrap a = 1;
  clockid_t __ob_wrap b = 2;
  pid_t __ob_wrap c = 4;
  _Static_assert(_Generic((a + b), int __ob_wrap: 1, default: 0), "a + b should be __ob_wrap int");
  _Static_assert(_Generic((a + c), pid_t __ob_wrap: 1, default: 0), "a + c should be __ob_wrap pid_t");
}

typedef unsigned long __ob_trap test_size_t;
typedef int __ob_wrap test_wrap_int;

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

void test_mixed_specifier_attribute() {
  int __ob_wrap __attribute__((overflow_behavior(wrap))) a; // expected-warning {{redundant overflow behavior specification; both specifier and attribute specify 'wrap'}}
  int __ob_trap __attribute__((overflow_behavior(trap))) b; // expected-warning {{redundant overflow behavior specification; both specifier and attribute specify 'trap'}}

  int __ob_wrap __attribute__((overflow_behavior(trap))) c; // expected-error {{conflicting overflow behavior specification; specifier specifies 'wrap' but attribute specifies 'trap'}}
  int __ob_trap __attribute__((overflow_behavior(wrap))) d; // expected-error {{conflicting overflow behavior specification; specifier specifies 'trap' but attribute specifies 'wrap'}}

  int __ob_wrap e; // OK
  int __attribute__((overflow_behavior(trap))) f; // OK
}

void test_incompatible_obt_assignment() {
  int __ob_trap trap_var;
  int __ob_wrap wrap_var;

  trap_var = wrap_var; // expected-error {{assigning to '__ob_trap int' from '__ob_wrap int' with incompatible overflow behavior types ('__ob_trap' and '__ob_wrap')}}
  wrap_var = trap_var; // expected-error {{assigning to '__ob_wrap int' from '__ob_trap int' with incompatible overflow behavior types ('__ob_wrap' and '__ob_trap')}}
}

void test_signedness_differences(long long __ob_trap a,
                                 unsigned long __ob_trap b) {
  (void)(a + b); // OK
}

typedef unsigned long __ob_trap size_t_trap;
void takes_three_ints(int, int, int);
void test_explicit_cast_to_function_arg(void) {
  // expected-warning@+1 {{passing argument of type 'size_t_trap' (aka '__ob_trap unsigned long') to parameter of type 'int' discards overflow behavior at function boundary}}
  takes_three_ints(0, 0, (size_t_trap)0);
}

_Bool test_obt_to_bool_conversion(size_t_trap a) {
  return a; // expected-warning {{implicit conversion from 'size_t_trap' (aka '__ob_trap unsigned long') to '_Bool' discards overflow behavior}}
}

void takes_obt_param(size_t_trap);
void test_int_literal_to_obt_param(void) {
  takes_obt_param(0);
  takes_obt_param(42);
}

// OBT specifiers cannot qualify pointers
void (* __ob_wrap bad_fp)(int); // expected-error {{__ob_wrap specifier cannot be applied to non-integer type 'void (*)(int)'}}
int (* __ob_wrap bad_arr_ptr)[10]; // expected-error {{__ob_wrap specifier cannot be applied to non-integer type 'int (*)[10]'}}
int * __ob_wrap bad_ptr; // expected-error {{__ob_wrap specifier cannot be applied to non-integer type 'int *'}}
int * __ob_trap bad_ptr2; // expected-error {{__ob_trap specifier cannot be applied to non-integer type 'int *'}}

// OBT specifiers on the base integer type (before *) are fine
int __ob_wrap *good_ptr;
int __ob_trap *good_ptr2;
