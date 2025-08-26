// RUN: %clang_cc1 %s -Winteger-overflow -Wno-unused-value -foverflow-behavior-types -Wconstant-conversion -verify -fsyntax-only

typedef int __attribute__((overflow_behavior)) bad_arg_count; // expected-error {{'overflow_behavior' attribute takes one argument}}
typedef int __attribute__((overflow_behavior(not_real))) bad_arg_spec; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef int __attribute__((overflow_behavior("not_real"))) bad_arg_spec_str; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef char* __attribute__((overflow_behavior("wrap"))) bad_type; // expected-warning {{'overflow_behavior' attribute cannot be applied to non-integer type 'char *'; attribute ignored}}

typedef int __attribute__((overflow_behavior(wrap))) ok_wrap; // OK
typedef long __attribute__((overflow_behavior(no_wrap))) ok_nowrap; // OK
typedef unsigned long __attribute__((overflow_behavior("wrap"))) str_ok_wrap; // OK
typedef char __attribute__((overflow_behavior("no_wrap"))) str_ok_nowrap; // OK

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __no_wrap __attribute__((overflow_behavior(no_wrap)))

struct struct_not_allowed {
  int i;
} __attribute__((overflow_behavior(wrap))); // expected-warning {{'overflow_behavior' attribute only applies to variables, typedefs, and non-static data members}}

void foo() {
  (2147483647 + 100); // expected-warning {{overflow in expression; result is }}
  (ok_wrap)2147483647 + 100; // no warn
}

// C++ stuff expects no warns
typedef int __attribute__((overflow_behavior(wrap))) wrap_int;

template <typename T>
T bar(T a) {
  return 1UL;
}

void f() {
  wrap_int a = 4;
  bar(a);
}

class TestOverload {
  public:
    void operator<<(int other); // expected-note {{candidate function}}
    void operator<<(char other); // expected-note {{candidate function}}
};

void test_overload1() {
  wrap_int a = 4;
  TestOverload TO;
  TO << a; // expected-error {{use of overloaded operator '<<' is ambiguous}}
}

// expected-note@+1 {{candidate function}}
int add_one(long a) { // expected-note {{candidate function}}
  return (a + 1);
}

// expected-note@+1 {{candidate function}}
int add_one(char a) { // expected-note {{candidate function}}
  return (a + 1);
}

// expected-note@+1 {{candidate function}}
int add_one(int a) { // expected-note {{candidate function}}
  return (a + 1);
}

void test_overload2(wrap_int a) {
  // to be clear, this is the same ambiguity expected when using a non-OBT int type.
  add_one(a); // expected-error {{call to 'add_one' is ambiguous}}
  long __attribute__((overflow_behavior(no_wrap))) b; // don't consider underlying type an exact match.
  add_one(b); // expected-error {{call to 'add_one' is ambiguous}}
}

void func(__no_wrap int i);
void func(int i); // Overload, not invalid redeclaration

// TODO: make this diagnostic message more descriptive
template <typename Ty>
void func2(__no_wrap Ty i) {} // expected-warning {{'overflow_behavior' attribute cannot be applied to non-integer type 'Ty'; attribute ignored}}

template <typename Ty>
struct S {};

template <>
struct S<__no_wrap int> {};

template <>
struct S<int> {};

void ptr(int a) {
  int __no_wrap *p = &a; // expected-error-re {{cannot initialize a variable of type '__no_wrap int *' {{.*}}with an rvalue of type 'int *'}}
}

void ptr2(__no_wrap int a) {
  int *p = &a; // expected-error-re {{cannot initialize a variable of type 'int *' {{.*}}with an rvalue of type '__no_wrap int *'}}
}

void overloadme(__no_wrap int a); // expected-note {{candidate function}}
void overloadme(short a); // expected-note {{candidate function}}

void test_overload_ambiguity() {
  int a;
  overloadme(a); // expected-error {{call to 'overloadme' is ambiguous}}
}

void f(void) __attribute__((overflow_behavior(wrap))); // expected-warning {{'overflow_behavior' attribute cannot be applied to non-integer type 'void (void)'; attribute ignored}}

typedef float __attribute__((overflow_behavior(wrap))) wrap_float; // expected-warning {{'overflow_behavior' attribute cannot be applied to non-integer type 'float'; attribute ignored}}

void pointer_compatibility(int* i_ptr) {
  __no_wrap int* nowrap_ptr;

  // static_cast should fail.
  nowrap_ptr = static_cast<__no_wrap int*>(i_ptr); // expected-error {{static_cast from 'int *' to '__no_wrap int *' is not allowed}}

  // reinterpret_cast should succeed.
  nowrap_ptr = reinterpret_cast<__no_wrap int*>(i_ptr);
  (void)nowrap_ptr;
}

void cpp_constexpr_bracket_initialization() {
  constexpr short cx1 = {(int __wrap)100000}; // expected-error {{constant expression evaluates to 100000 which cannot be narrowed to type 'short'}}
  // expected-warning@-1 {{implicit conversion from '__wrap int' to 'const short' changes value from 100000 to -31072}}
  // expected-note@-2 {{insert an explicit cast to silence this issue}}

  constexpr short __wrap cx2 = {(int)100000}; // OK - wrapping destination

  constexpr short __no_wrap cx3 = {(int)100000}; // expected-error {{constant expression evaluates to 100000 which cannot be narrowed to type '__no_wrap short'}}
  // expected-warning@-1 {{implicit conversion from 'int' to '__no_wrap short const' changes value from 100000 to -31072}}

  constexpr short cx4 = {(int __no_wrap)100000}; // expected-error {{constant expression evaluates to 100000 which cannot be narrowed to type 'short'}}
  // expected-warning@-1 {{implicit conversion from '__no_wrap int' to 'const short' changes value from 100000 to -31072}}
  // expected-note@-2 {{insert an explicit cast to silence this issue}}
}

// ensure that all qualifier placements result in the same canonical type
void test_qualifier_placements() {
  using ConstInt = const int;
  using WrapConstInt1 = ConstInt __attribute__((overflow_behavior(wrap)));
  using WrapConstInt2 = const int __attribute__((overflow_behavior(wrap)));
  typedef const int __wrap const_int_wrap;
  typedef int __wrap const int_wrap_const;
  typedef int __no_wrap const int_no_wrap_const;

  static_assert(__is_same(WrapConstInt1, WrapConstInt2));
  static_assert(__is_same(const_int_wrap, int_wrap_const));
  static_assert(!__is_same(const_int_wrap, int_no_wrap_const));
}
