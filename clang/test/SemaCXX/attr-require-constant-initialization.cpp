// RUN: %clang_cc1 -fsyntax-only -verify -DTEST_ONE -std=c++03 %s
// RUN: %clang_cc1 -fsyntax-only -verify -DTEST_ONE -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -DTEST_ONE -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -DTEST_TWO \
// RUN: -Wglobal-constructors -std=c++14 %s
// RUN: %clang_cc1 -fsyntax-only -verify -DTEST_THREE -xc %s

#define ATTR __attribute__((require_constant_initialization)) // expected-note 0+ {{expanded from macro}}

int ReturnInt(); // expected-note 0+ {{declared here}}

struct PODType { // expected-note 0+ {{declared here}}
  int value;
  int value2;
};

#if defined(__cplusplus)

#if __cplusplus >= 201103L
struct LitType {
  constexpr LitType() : value(0) {}
  constexpr LitType(int x) : value(x) {}
  LitType(void *) : value(-1) {} // expected-note 0+ {{declared here}}
  int value;
};
#endif

struct NonLit { // expected-note 0+ {{declared here}}
#if __cplusplus >= 201402L
  constexpr NonLit() : value(0) {}
  constexpr NonLit(int x) : value(x) {}
#else
  NonLit() : value(0) {} // expected-note 0+ {{declared here}}
  NonLit(int x) : value(x) {}
#endif
  NonLit(void *) : value(-1) {} // expected-note 0+ {{declared here}}
  ~NonLit() {}
  int value;
};

struct StoresNonLit {
#if __cplusplus >= 201402L
  constexpr StoresNonLit() : obj() {}
  constexpr StoresNonLit(int x) : obj(x) {}
#else
  StoresNonLit() : obj() {} // expected-note 0+ {{declared here}}
  StoresNonLit(int x) : obj(x) {}
#endif
  StoresNonLit(void *p) : obj(p) {}
  NonLit obj;
};

#endif // __cplusplus


#if defined(TEST_ONE) // Test semantics of attribute

// Test diagnostics when attribute is applied to non-static declarations.
void test_func_local(ATTR int param) { // expected-error {{only applies to global variables}}
  ATTR int x = 42;                     // expected-error {{only applies to}}
  ATTR extern int y;
}
struct ATTR class_mem { // expected-error {{only applies to}}
  ATTR int x;           // expected-error {{only applies to}}
};

// [basic.start.static]p2.1
// if each full-expression (including implicit conversions) that appears in
// the initializer of a reference with static or thread storage duration is
// a constant expression (5.20) and the reference is bound to a glvalue
// designating an object with static storage duration, to a temporary object
// (see 12.2) or subobject thereof, or to a function;

// Test binding to a static glvalue
const int glvalue_int = 42;
const int glvalue_int2 = ReturnInt();
ATTR const int &glvalue_ref ATTR = glvalue_int;
ATTR const int &glvalue_ref2 ATTR = glvalue_int2;
ATTR __thread const int &glvalue_ref_tl = glvalue_int;

void test_basic_start_static_2_1() {
  const int non_global = 42;
  ATTR static const int &local_init = non_global; // expected-error {{variable does not have a constant initializer}}
  // expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
#if __cplusplus >= 201103L
  // expected-note@-3 {{reference to 'non_global' is not a constant expression}}
  // expected-note@-5 {{declared here}}
#else
  // expected-note@-6 {{subexpression not valid in a constant expression}}
#endif
  ATTR static const int &global_init = glvalue_int;
  ATTR static const int &temp_init = 42;
}

ATTR const int &temp_ref = 42;
ATTR const int &temp_ref2 = ReturnInt(); // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
#if __cplusplus >= 201103L
// expected-note@-3 {{non-constexpr function 'ReturnInt' cannot be used in a constant expression}}
#else
// expected-note@-5 {{subexpression not valid in a constant expression}}
#endif
ATTR const NonLit &nl_temp_ref = 42; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
#if __cplusplus >= 201103L
// expected-note@-3 {{non-literal type 'const NonLit' cannot be used in a constant expression}}
#else
// expected-note@-5 {{subexpression not valid in a constant expression}}
#endif

#if __cplusplus >= 201103L
ATTR const LitType &lit_temp_ref = 42;
ATTR const int &subobj_ref = LitType{}.value;
#endif

ATTR const int &nl_subobj_ref = NonLit().value; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
#if __cplusplus >= 201103L
// expected-note-re@-3 {{non-literal type '{{.*}}' cannot be used in a constant expression}}
#else
// expected-note@-5 {{subexpression not valid in a constant expression}}
#endif

struct TT1 {
  ATTR static const int &no_init;
  ATTR static const int &glvalue_init;
  ATTR static const int &temp_init;
  ATTR static const int &subobj_init;
#if __cplusplus >= 201103L
  ATTR static thread_local const int &tl_glvalue_init;
  ATTR static thread_local const int &tl_temp_init; // expected-note {{required by 'require_constant_initialization' attribute here}}
#endif
};
const int &TT1::glvalue_init = glvalue_int;
const int &TT1::temp_init = 42;
const int &TT1::subobj_init = PODType().value;
#if __cplusplus >= 201103L
thread_local const int &TT1::tl_glvalue_init = glvalue_int;
thread_local const int &TT1::tl_temp_init = 42; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{reference to temporary is not a constant expression}}
// expected-note@-2 {{temporary created here}}
#endif

// [basic.start.static]p2.2
// if an object with static or thread storage duration is initialized by a
// constructor call, and if the initialization full-expression is a constant
// initializer for the object;

void test_basic_start_static_2_2() {
#if __cplusplus < 201103L
  ATTR static PODType pod;
#else
  ATTR static PODType pod; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-constexpr constructor 'PODType' cannot be used in a constant expression}}
#endif
  ATTR static PODType pot2 = {ReturnInt()}; // expected-error {{variable does not have a constant initializer}}
                                            // expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
#if __cplusplus >= 201103L
// expected-note@-3 {{non-constexpr function 'ReturnInt' cannot be used in a constant expression}}
#else
// expected-note@-5 {{subexpression not valid in a constant expression}}
#endif

#if __cplusplus >= 201103L
  constexpr LitType l;
  ATTR static LitType static_lit = l;
  ATTR static LitType static_lit2 = (void *)0; // expected-error {{variable does not have a constant initializer}}
  // expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
  // expected-note@-2 {{non-constexpr constructor 'LitType' cannot be used in a constant expression}}
  ATTR static LitType static_lit3 = ReturnInt(); // expected-error {{variable does not have a constant initializer}}
  // expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
  // expected-note@-2 {{non-constexpr function 'ReturnInt' cannot be used in a constant expression}}
  ATTR thread_local LitType tls = 42;
#endif
}

struct TT2 {
  ATTR static PODType pod_noinit;
#if __cplusplus >= 201103L
// expected-note@-2 {{required by 'require_constant_initialization' attribute here}}
#endif
  ATTR static PODType pod_copy_init; // expected-note {{required by 'require_constant_initialization' attribute here}}
#if __cplusplus >= 201402L
  ATTR static constexpr LitType lit = {};
  ATTR static const NonLit non_lit;
  ATTR static const NonLit non_lit_list_init;
  ATTR static const NonLit non_lit_copy_init;
#endif
};
PODType TT2::pod_noinit; // expected-note 0+ {{declared here}}
#if __cplusplus >= 201103L
// expected-error@-2 {{variable does not have a constant initializer}}
// expected-note@-3 {{non-constexpr constructor 'PODType' cannot be used in a constant expression}}
#endif
PODType TT2::pod_copy_init(TT2::pod_noinit); // expected-error {{variable does not have a constant initializer}}
#if __cplusplus >= 201103L
// expected-note@-2 {{read of non-constexpr variable 'pod_noinit' is not allowed in a constant expression}}
// expected-note@-3 {{in call to 'PODType(pod_noinit)'}}
#else
// expected-note@-5 {{subexpression not valid in a constant expression}}
#endif
#if __cplusplus >= 201402L
const NonLit TT2::non_lit(42);
const NonLit TT2::non_lit_list_init = {42};
// FIXME: This is invalid, but we incorrectly elide the copy. It's OK if we
// start diagnosing this.
const NonLit TT2::non_lit_copy_init = 42;
#endif

#if __cplusplus >= 201103L
ATTR LitType lit_ctor;
ATTR LitType lit_ctor2{};
ATTR LitType lit_ctor3 = {};
ATTR __thread LitType lit_ctor_tl = {};

#if __cplusplus >= 201402L
ATTR NonLit nl_ctor;
ATTR NonLit nl_ctor2{};
ATTR NonLit nl_ctor3 = {};
ATTR thread_local NonLit nl_ctor_tl = {};
ATTR StoresNonLit snl;
#else
ATTR NonLit nl_ctor; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-constexpr constructor 'NonLit' cannot be used in a constant expression}}
ATTR NonLit nl_ctor2{}; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-constexpr constructor 'NonLit' cannot be used in a constant expression}}
ATTR NonLit nl_ctor3 = {}; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-constexpr constructor 'NonLit' cannot be used in a constant expression}}
ATTR thread_local NonLit nl_ctor_tl = {}; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-constexpr constructor 'NonLit' cannot be used in a constant expression}}
ATTR StoresNonLit snl; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-constexpr constructor 'StoresNonLit' cannot be used in a constant expression}}
#endif

// Non-literal types cannot appear in the initializer of a non-literal type.
ATTR int nl_in_init = NonLit{42}.value; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-literal type 'NonLit' cannot be used in a constant expression}}
ATTR int lit_in_init = LitType{42}.value;
#endif

// [basic.start.static]p2.3
// if an object with static or thread storage duration is not initialized by a
// constructor call and if either the object is value-initialized or every
// full-expression that appears in its initializer is a constant expression.
void test_basic_start_static_2_3() {
  ATTR static int static_local = 42;
  ATTR static int static_local2; // zero-initialization takes place
#if __cplusplus >= 201103L
  ATTR thread_local int tl_local = 42;
#endif
}

ATTR int no_init; // zero initialization takes place
ATTR int arg_init = 42;
ATTR PODType pod_init = {};
ATTR PODType pod_missing_init = {42 /* should have second arg */};
ATTR PODType pod_full_init = {1, 2};
ATTR PODType pod_non_constexpr_init = {1, ReturnInt()}; // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
#if __cplusplus >= 201103L
// expected-note@-3 {{non-constexpr function 'ReturnInt' cannot be used in a constant expression}}
#else
// expected-note@-5 {{subexpression not valid in a constant expression}}
#endif

#if __cplusplus >= 201103L
ATTR int val_init{};
ATTR int brace_init = {};
#endif

ATTR __thread int tl_init = 0;
typedef const char *StrType;

#if __cplusplus >= 201103L

// Test that the validity of the selected constructor is checked, not just the
// initializer
struct NotC {
  constexpr NotC(void *) {}
  NotC(int) {} // expected-note 0+ {{declared here}}
};
template <class T>
struct TestCtor {
  constexpr TestCtor(int x) : value(x) {}
  // expected-note@-1  {{non-constexpr constructor 'NotC' cannot be used in a constant expression}}
  T value;
};
ATTR TestCtor<NotC> t(42); // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{in call to 'TestCtor(42)'}}
#endif

// Test various array types
ATTR const char *foo[] = {"abc", "def"};
ATTR PODType bar[] = {{}, {123, 456}};


namespace AttrAddedTooLate {
  struct A {
    static const int n = 0; // expected-note {{here}}
  };
  ATTR const int A::n; // expected-warning {{added after initialization}}

  int m = 0; // expected-note {{here}}
  extern ATTR int m; // expected-warning {{added after initialization}}
}

#elif defined(TEST_TWO) // Test for duplicate warnings
struct NotC {
  constexpr NotC(void *) {}
  NotC(int) {} // expected-note 2 {{declared here}}
};
template <class T>
struct TestCtor {
  constexpr TestCtor(int x) : value(x) {} // expected-note 2 {{non-constexpr constructor}}
  T value;
};

ATTR LitType non_const_lit(nullptr); // expected-error {{variable does not have a constant initializer}}
// expected-note@-1 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-2 {{non-constexpr constructor 'LitType' cannot be used in a constant expression}}
ATTR NonLit non_const(nullptr); // expected-error {{variable does not have a constant initializer}}
// expected-warning@-1 {{declaration requires a global destructor}}
// expected-note@-2 {{required by 'require_constant_initialization' attribute here}}
// expected-note@-3 {{non-constexpr constructor 'NonLit' cannot be used in a constant expression}}
LitType const_init_lit(nullptr);              // expected-warning {{declaration requires a global constructor}}
NonLit const_init{42};                        // expected-warning {{declaration requires a global destructor}}
constexpr TestCtor<NotC> inval_constexpr(42); // expected-error {{must be initialized by a constant expression}}
// expected-note@-1 {{in call to 'TestCtor(42)'}}
ATTR constexpr TestCtor<NotC> inval_constexpr2(42); // expected-error {{must be initialized by a constant expression}}
// expected-note@-1 {{in call to 'TestCtor(42)'}}

#elif defined(TEST_THREE)
#if defined(__cplusplus)
#error This test requires C
#endif
// Test that using the attribute in C results in a diagnostic
ATTR int x = 0; // expected-warning {{attribute ignored}}
#else
#error No test case specified
#endif // defined(TEST_N)
