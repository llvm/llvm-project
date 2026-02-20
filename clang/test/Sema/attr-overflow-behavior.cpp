// RUN: %clang_cc1 %s -Winteger-overflow -Wno-unused-value -fexperimental-overflow-behavior-types -Wconstant-conversion -Woverflow-behavior-conversion -verify -fsyntax-only

typedef int __attribute__((overflow_behavior)) bad_arg_count; // expected-error {{'overflow_behavior' attribute takes one argument}}
typedef int __attribute__((overflow_behavior(not_real))) bad_arg_spec; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef int __attribute__((overflow_behavior("not_real"))) bad_arg_spec_str; // expected-error {{'not_real' is not a valid argument to attribute 'overflow_behavior'}}
typedef char* __attribute__((overflow_behavior("wrap"))) bad_type; // expected-error {{'overflow_behavior' attribute cannot be applied to non-integer type 'char *'}}

typedef int __attribute__((overflow_behavior(wrap))) ok_wrap; // OK
typedef long __attribute__((overflow_behavior(trap))) ok_nowrap; // OK
typedef unsigned long __attribute__((overflow_behavior("wrap"))) str_ok_wrap; // OK
typedef char __attribute__((overflow_behavior("trap"))) str_ok_nowrap; // OK

#define __wrap __attribute__((overflow_behavior(wrap)))
#define __trap __attribute__((overflow_behavior(trap)))

struct struct_not_allowed {
  int i;
} __attribute__((overflow_behavior(wrap))); // expected-warning {{'overflow_behavior' attribute only applies to variables, typedefs, and data members}}

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
  long __attribute__((overflow_behavior(trap))) b; // don't consider underlying type an exact match.
  add_one(b); // expected-error {{call to 'add_one' is ambiguous}}
}

void func(__ob_trap int i);
void func(int i); // Overload, not invalid redeclaration

template <typename Ty>
void func2(__ob_trap Ty i) {} // expected-error {{__ob_trap specifier cannot be applied to non-integer type 'Ty'}}

template <typename Ty>
struct S {};

template <>
struct S<__ob_trap int> {};

template <>
struct S<int> {};

void ptr(int a) {
  int __ob_trap *p = &a; // expected-error-re {{cannot initialize a variable of type '__ob_trap int *' {{.*}}with an rvalue of type 'int *'}}
}

void ptr2(__ob_trap int a) {
  int *p = &a; // expected-error-re {{cannot initialize a variable of type 'int *' {{.*}}with an rvalue of type '__ob_trap int *'}}
}

void overloadme(__ob_trap int a); // expected-note {{candidate function}}
void overloadme(short a); // expected-note {{candidate function}}

void test_overload_ambiguity() {
  int a;
  overloadme(a); // expected-error {{call to 'overloadme' is ambiguous}}
}

void f(void) __attribute__((overflow_behavior(wrap))); // expected-error {{'overflow_behavior' attribute cannot be applied to non-integer type 'void (void)'}}

typedef float __attribute__((overflow_behavior(wrap))) wrap_float; // expected-error {{'overflow_behavior' attribute cannot be applied to non-integer type 'float'}}

void pointer_compatibility(int* i_ptr) {
  __ob_trap int* nowrap_ptr;

  // static_cast should fail.
  nowrap_ptr = static_cast<__ob_trap int*>(i_ptr); // expected-error {{static_cast from 'int *' to '__ob_trap int *' is not allowed}}

  // reinterpret_cast should succeed.
  nowrap_ptr = reinterpret_cast<__ob_trap int*>(i_ptr);
  (void)nowrap_ptr;
}

void cpp_constexpr_bracket_initialization() {
  constexpr short cx1 = {(int __ob_wrap)100000}; // expected-error {{constant expression evaluates to 100000 which cannot be narrowed to type 'short'}}
  // expected-note@-1 {{insert an explicit cast to silence this issue}}

  constexpr short __ob_wrap cx2 = {100000}; // expected-error {{constant expression evaluates to 100000 which cannot be narrowed to type '__ob_wrap short'}}

  constexpr short __ob_trap cx3 = {(int)100000}; // expected-error {{constant expression evaluates to 100000 which cannot be narrowed to type '__ob_trap short'}}
  // expected-warning@-1 {{implicit conversion from 'int' to '__ob_trap short const' changes value from 100000 to -31072}}

  constexpr short cx4 = {(int __ob_trap)100000}; // expected-error {{constant expression evaluates to 100000 which cannot be narrowed to type 'short'}}
  // expected-warning@-1 {{implicit conversion from '__ob_trap int' to '__ob_trap const short' changes value from 100000 to -31072}}
  // expected-note@-2 {{insert an explicit cast to silence this issue}}
}

// ensure that all qualifier placements result in the same canonical type
void test_qualifier_placements() {
  using ConstInt = const int;
  using WrapConstInt1 = ConstInt __attribute__((overflow_behavior(wrap)));
  using WrapConstInt2 = const int __attribute__((overflow_behavior(wrap)));
  typedef const int __ob_wrap const_int_wrap;
  typedef int __ob_wrap const int_wrap_const;
  typedef int __ob_trap const int_trap_const;

  static_assert(__is_same(WrapConstInt1, WrapConstInt2));
  static_assert(__is_same(const_int_wrap, int_wrap_const));
  static_assert(!__is_same(const_int_wrap, int_trap_const));
}

void test_mixed_specifier_attribute() {
  int __ob_wrap __attribute__((overflow_behavior(wrap))) a; // expected-warning {{redundant overflow behavior specification; both specifier and attribute specify 'wrap'}}
  int __ob_trap __attribute__((overflow_behavior(trap))) b; // expected-warning {{redundant overflow behavior specification; both specifier and attribute specify 'trap'}}

  int __ob_wrap __attribute__((overflow_behavior(trap))) c; // expected-error {{conflicting overflow behavior specification; specifier specifies 'wrap' but attribute specifies 'trap'}}
  int __ob_trap __attribute__((overflow_behavior(wrap))) d; // expected-error {{conflicting overflow behavior specification; specifier specifies 'trap' but attribute specifies 'wrap'}}

  int __ob_wrap e; // OK
  int __attribute__((overflow_behavior(trap))) f; // OK
}

void test_incompatible_obt_initialization() {
  int __ob_trap a = 10;
  int __ob_wrap b = 20;

  int __ob_wrap c = a; // expected-error {{cannot initialize a variable of type '__ob_wrap int' with an lvalue of type '__ob_trap int'}}
  int __ob_trap d = b; // expected-error {{cannot initialize a variable of type '__ob_trap int' with an lvalue of type '__ob_wrap int'}}
}

void constant_conversion() {
  short x1 = (int __ob_wrap)100000;
  short __ob_wrap x2 = (int)100000; // No warning expected
  // expected-warning@+1 {{implicit conversion from 'int' to '__ob_trap short' changes value from 100000 to -31072}}
  short __ob_trap x3 = (int)100000;
  // expected-warning@+1 {{implicit conversion from '__ob_trap int' to '__ob_trap short' changes value from 100000 to -31072}}
  short x4 = (int __ob_trap)100000;

  unsigned short __ob_wrap ux1 = (unsigned int)100000; // No warning - wrapping expected
  unsigned short ux2 = (unsigned int __ob_wrap)100000;
  unsigned short __ob_trap ux3 = (unsigned int)100000; // expected-warning {{implicit conversion from 'unsigned int' to '__ob_trap unsigned short' changes value from 100000 to 34464}}
  unsigned short __ob_trap ux4 = (unsigned int __ob_trap)100000; // expected-warning {{implicit conversion from '__ob_trap unsigned int' to '__ob_trap unsigned short' changes value from 100000 to 34464}}
  unsigned short __ob_trap ux5 = (unsigned int __ob_wrap)100000; // expected-error {{assigning to '__ob_trap unsigned short' from '__ob_wrap unsigned int' with incompatible overflow behavior types ('__ob_trap' and '__ob_wrap')}}
}

// OBT on data members (both static and non-static)
struct DataMembers {
  __ob_wrap int nonstatic_specifier;
  int __attribute__((overflow_behavior(wrap))) nonstatic_attr;
  static __ob_trap int static_specifier;
  static int __ob_wrap static_specifier2;
  static int __attribute__((overflow_behavior(trap))) static_attr;
};
__ob_trap int DataMembers::static_specifier = 0;
int __ob_wrap DataMembers::static_specifier2 = 0;
int __attribute__((overflow_behavior(trap))) DataMembers::static_attr = 0;
