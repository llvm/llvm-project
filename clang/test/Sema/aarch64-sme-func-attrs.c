// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme2 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme2 -fsyntax-only -verify=expected-cpp -x c++ %s

// Valid attributes

void sme_arm_streaming(void) __arm_streaming;
void sme_arm_streaming_compatible(void) __arm_streaming_compatible;

__arm_new("za") void sme_arm_new_za(void) {}
void sme_arm_shared_za(void) __arm_inout("za");
void sme_arm_preserves_za(void) __arm_preserves("za");

__arm_new("za") void sme_arm_streaming_new_za(void) __arm_streaming {}
void sme_arm_streaming_shared_za(void) __arm_streaming __arm_inout("za");
void sme_arm_streaming_preserves_za(void) __arm_streaming __arm_preserves("za");

__arm_new("za") void sme_arm_sc_new_za(void) __arm_streaming_compatible {}
void sme_arm_sc_shared_za(void) __arm_streaming_compatible __arm_inout("za");
void sme_arm_sc_preserves_za(void) __arm_streaming_compatible __arm_preserves("za");

__arm_locally_streaming void sme_arm_locally_streaming(void) { }
__arm_locally_streaming void sme_arm_streaming_and_locally_streaming(void) __arm_streaming { }
__arm_locally_streaming void sme_arm_streaming_and_streaming_compatible(void) __arm_streaming_compatible { }

__arm_locally_streaming __arm_new("za") void sme_arm_ls_new_za(void) { }
__arm_locally_streaming void sme_arm_ls_shared_za(void) __arm_inout("za") { }
__arm_locally_streaming void sme_arm_ls_preserves_za(void) __arm_preserves("za") { }

// Valid attributes on function pointers

void streaming_ptr(void) __arm_streaming;
typedef  void (*fptrty1) (void) __arm_streaming;
fptrty1 call_streaming_func() { return streaming_ptr; }

void streaming_compatible_ptr(void) __arm_streaming_compatible;
typedef void (*fptrty2) (void) __arm_streaming_compatible;
fptrty2 call_sc_func() { return streaming_compatible_ptr; }

void shared_za_ptr(void) __arm_inout("za");
typedef void (*fptrty3) (void) __arm_inout("za");
fptrty3 call_shared_za_func() { return shared_za_ptr; }

void preserves_za_ptr(void) __arm_preserves("za");
typedef void (*fptrty4) (void) __arm_preserves("za");
fptrty4 call_preserve_za_func() { return preserves_za_ptr; }

typedef void (*fptrty6) (void);
fptrty6 cast_nza_func_to_normal() { return sme_arm_new_za; }
fptrty6 cast_ls_func_to_normal() { return sme_arm_locally_streaming; }

// Invalid attributes

// expected-cpp-error@+4 {{'__arm_streaming_compatible' and '__arm_streaming' are not compatible}}
// expected-cpp-note@+3 {{conflicting attribute is here}}
// expected-error@+2 {{'__arm_streaming_compatible' and '__arm_streaming' are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void streaming_mode(void) __arm_streaming __arm_streaming_compatible;

// expected-cpp-error@+4 {{'__arm_streaming' and '__arm_streaming_compatible' are not compatible}}
// expected-cpp-note@+3 {{conflicting attribute is here}}
// expected-error@+2 {{'__arm_streaming' and '__arm_streaming_compatible' are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void streaming_compatible(void) __arm_streaming_compatible __arm_streaming;

// expected-cpp-error@+2 {{'__arm_new("za")' and '__arm_inout("za")' are not compatible}}
// expected-error@+1 {{'__arm_new("za")' and '__arm_inout("za")' are not compatible}}
__arm_new("za") void new_shared_za(void) __arm_inout("za") {}

// expected-cpp-error@+2 {{'__arm_new("za")' and '__arm_preserves("za")' are not compatible}}
// expected-error@+1 {{'__arm_new("za")' and '__arm_preserves("za")' are not compatible}}
__arm_new("za") void new_preserves_za(void) __arm_preserves("za") {}

// Invalid attributes on function pointers

// expected-cpp-error@+4 {{'__arm_streaming_compatible' and '__arm_streaming' are not compatible}}
// expected-cpp-note@+3 {{conflicting attribute is here}}
// expected-error@+2 {{'__arm_streaming_compatible' and '__arm_streaming' are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
void streaming_ptr_invalid(void) __arm_streaming __arm_streaming_compatible;
// expected-cpp-error@+4 {{'__arm_streaming_compatible' and '__arm_streaming' are not compatible}}
// expected-cpp-note@+3 {{conflicting attribute is here}}
// expected-error@+2 {{'__arm_streaming_compatible' and '__arm_streaming' are not compatible}}
// expected-note@+1 {{conflicting attribute is here}}
typedef void (*fptrty7) (void) __arm_streaming __arm_streaming_compatible;
fptrty7 invalid_streaming_func() { return streaming_ptr_invalid; }

// expected-warning@+2 {{'__arm_streaming' only applies to non-K&R-style functions}}
// expected-error@+1 {{'__arm_streaming' only applies to function types; type here is 'void ()'}}
void function_no_prototype() __arm_streaming;

//
// Check for incorrect conversions of function pointers with the attributes
//

typedef void (*n_ptrty) (void);
typedef void (*s_ptrty) (void) __arm_streaming;
s_ptrty return_valid_streaming_fptr(s_ptrty f) { return f; }

// expected-cpp-error@+2 {{cannot initialize return object of type 's_ptrty' (aka 'void (*)() __arm_streaming') with an lvalue of type 'n_ptrty' (aka 'void (*)()')}}
// expected-error@+1 {{incompatible function pointer types returning 'n_ptrty' (aka 'void (*)(void)') from a function with result type 's_ptrty' (aka 'void (*)(void) __arm_streaming')}}
s_ptrty return_invalid_fptr_streaming_normal(n_ptrty f) { return f; }
// expected-cpp-error@+2 {{cannot initialize return object of type 'n_ptrty' (aka 'void (*)()') with an lvalue of type 's_ptrty' (aka 'void (*)() __arm_streaming')}}
// expected-error@+1 {{incompatible function pointer types returning 's_ptrty' (aka 'void (*)(void) __arm_streaming') from a function with result type 'n_ptrty' (aka 'void (*)(void)')}}
n_ptrty return_invalid_fptr_normal_streaming(s_ptrty f) { return f; }

// Test an instance where the result type is not a prototyped function, such that we still get a diagnostic.
typedef void (*nonproto_n_ptrty) ();
// expected-cpp-error@+2 {{cannot initialize return object of type 'nonproto_n_ptrty' (aka 'void (*)()') with an lvalue of type 's_ptrty' (aka 'void (*)() __arm_streaming')}}
// expected-error@+1 {{incompatible function pointer types returning 's_ptrty' (aka 'void (*)(void) __arm_streaming') from a function with result type 'nonproto_n_ptrty' (aka 'void (*)()')}}
nonproto_n_ptrty return_invalid_fptr_streaming_nonprotonormal(s_ptrty f) { return f; }

typedef void (*sc_ptrty) (void) __arm_streaming_compatible;
sc_ptrty return_valid_streaming_compatible_fptr(sc_ptrty f) { return f; }

// expected-cpp-error@+2 {{cannot initialize return object of type 'sc_ptrty' (aka 'void (*)() __arm_streaming_compatible') with an lvalue of type 'n_ptrty' (aka 'void (*)()')}}
// expected-error@+1 {{incompatible function pointer types returning 'n_ptrty' (aka 'void (*)(void)') from a function with result type 'sc_ptrty' (aka 'void (*)(void) __arm_streaming_compatible')}}
sc_ptrty return_invalid_fptr_streaming_compatible_normal(n_ptrty f) { return f; }
// expected-cpp-error@+2 {{cannot initialize return object of type 'n_ptrty' (aka 'void (*)()') with an lvalue of type 'sc_ptrty' (aka 'void (*)() __arm_streaming_compatible')}}
// expected-error@+1 {{incompatible function pointer types returning 'sc_ptrty' (aka 'void (*)(void) __arm_streaming_compatible') from a function with result type 'n_ptrty' (aka 'void (*)(void)')}}
n_ptrty return_invalid_fptr_normal_streaming_compatible(sc_ptrty f) { return f; }

typedef void (*sz_ptrty) (void) __arm_inout("za");
sz_ptrty return_valid_shared_za_fptr(sz_ptrty f) { return f; }


// expected-cpp-error@+2 {{cannot initialize return object of type 'sz_ptrty' (aka 'void (*)() __arm_inout("za")') with an lvalue of type 'n_ptrty' (aka 'void (*)()')}}
// expected-error@+1 {{incompatible function pointer types returning 'n_ptrty' (aka 'void (*)(void)') from a function with result type 'sz_ptrty' (aka 'void (*)(void) __arm_inout("za")')}}
sz_ptrty return_invalid_fptr_shared_za_normal(n_ptrty f) { return f; }
// expected-cpp-error@+2 {{cannot initialize return object of type 'n_ptrty' (aka 'void (*)()') with an lvalue of type 'sz_ptrty' (aka 'void (*)() __arm_inout("za")')}}
// expected-error@+1 {{incompatible function pointer types returning 'sz_ptrty' (aka 'void (*)(void) __arm_inout("za")') from a function with result type 'n_ptrty' (aka 'void (*)(void)')}}
n_ptrty return_invalid_fptr_normal_shared_za(sz_ptrty f) { return f; }

typedef void (*pz_ptrty) (void) __arm_preserves("za");
pz_ptrty return_valid_preserves_za_fptr(pz_ptrty f) { return f; }

// expected-cpp-error@+2 {{cannot initialize return object of type 'pz_ptrty' (aka 'void (*)() __arm_preserves("za")') with an lvalue of type 'n_ptrty' (aka 'void (*)()')}}
// expected-error@+1 {{incompatible function pointer types returning 'n_ptrty' (aka 'void (*)(void)') from a function with result type 'pz_ptrty' (aka 'void (*)(void) __arm_preserves("za")')}}
pz_ptrty return_invalid_fptr_preserves_za_normal(n_ptrty f) { return f; }
// expected-cpp-error@+2 {{cannot initialize return object of type 'n_ptrty' (aka 'void (*)()') with an lvalue of type 'pz_ptrty' (aka 'void (*)() __arm_preserves("za")')}}
// expected-error@+1 {{incompatible function pointer types returning 'pz_ptrty' (aka 'void (*)(void) __arm_preserves("za")') from a function with result type 'n_ptrty' (aka 'void (*)(void)')}}
n_ptrty return_invalid_fptr_normal_preserves_za(pz_ptrty f) { return f; }

// Test template instantiations
#ifdef __cplusplus
template <typename T> T templated(T x) __arm_streaming { return x; }
template <> int templated<int>(int x) __arm_streaming { return x + 1; }
template <> float templated<float>(float x) __arm_streaming { return x + 2; }
// expected-cpp-error@+2 {{explicit instantiation of 'templated' does not refer to a function template, variable template, member function, member class, or static data member}}
// expected-cpp-note@-4 {{candidate template ignored: could not match 'short (short) __arm_streaming' against 'short (short)'}}
template short templated<short>(short);
#endif

// Conflicting attributes on redeclarations

// expected-error@+5 {{function declared 'void (void) __arm_streaming_compatible' was previously declared 'void (void) __arm_streaming', which has different SME function attributes}}
// expected-note@+3 {{previous declaration is here}}
// expected-cpp-error@+3 {{function declared 'void () __arm_streaming_compatible' was previously declared 'void () __arm_streaming', which has different SME function attributes}}
// expected-cpp-note@+1 {{previous declaration is here}}
void redecl(void) __arm_streaming;
void redecl(void) __arm_streaming_compatible { }

// expected-error@+5 {{function declared 'void (void)' was previously declared 'void (void) __arm_preserves("za")', which has different SME function attributes}}
// expected-note@+3 {{previous declaration is here}}
// expected-cpp-error@+3 {{function declared 'void ()' was previously declared 'void () __arm_preserves("za")', which has different SME function attributes}}
// expected-cpp-note@+1 {{previous declaration is here}}
void redecl_preserve_za(void) __arm_preserves("za");;
void redecl_preserve_za(void) {}

// expected-error@+5 {{function declared 'void (void) __arm_preserves("za")' was previously declared 'void (void)', which has different SME function attributes}}
// expected-note@+3 {{previous declaration is here}}
// expected-cpp-error@+3 {{function declared 'void () __arm_preserves("za")' was previously declared 'void ()', which has different SME function attributes}}
// expected-cpp-note@+1 {{previous declaration is here}}
void redecl_nopreserve_za(void);
void redecl_nopreserve_za(void) __arm_preserves("za") {}

void non_za_definition(void (*shared_za_fn_ptr)(void) __arm_inout("za"), void (*preserves_za_fn_ptr)(void) __arm_preserves("za")) {
  sme_arm_new_za(); // OK
  // expected-error@+2 {{call to a shared ZA function requires the caller to have ZA state}}
  // expected-cpp-error@+1 {{call to a shared ZA function requires the caller to have ZA state}}
  sme_arm_shared_za();
  // expected-error@+2 {{call to a shared ZA function requires the caller to have ZA state}}
  // expected-cpp-error@+1 {{call to a shared ZA function requires the caller to have ZA state}}
  shared_za_fn_ptr();
  // expected-error@+2 {{call to a shared ZA function requires the caller to have ZA state}}
  // expected-cpp-error@+1 {{call to a shared ZA function requires the caller to have ZA state}}
  preserves_za_fn_ptr();
}

void shared_za_definition(void (*shared_za_fn_ptr)(void) __arm_inout("za")) __arm_inout("za") {
  sme_arm_shared_za(); // OK
  shared_za_fn_ptr(); // OK
}

__arm_new("za") void new_za_definition(void (*shared_za_fn_ptr)(void) __arm_inout("za")) {
  sme_arm_shared_za(); // OK
  shared_za_fn_ptr(); // OK
}

#ifdef __cplusplus
int shared_za_initializer(void) __arm_inout("za");
// expected-cpp-error@+1 {{call to a shared ZA function requires the caller to have ZA state}}
int global = shared_za_initializer();

struct S {
  virtual void shared_za_memberfn(void) __arm_inout("za");
};

struct S2 : public S {
// expected-cpp-error@+2 {{virtual function 'shared_za_memberfn' has different attributes ('void ()') than the function it overrides (which has 'void () __arm_inout("za")')}}
// expected-cpp-note@-5 {{overridden virtual function is here}}
  __arm_new("za") void shared_za_memberfn(void) override {}
};

// The '__arm_preserves("za")' property cannot be dropped when overriding a virtual
// function. It is however fine for the overriding function to be '__arm_preserves("za")'
// even though the function that it overrides is not.

struct S_PreservesZA {
  virtual void memberfn(void) __arm_preserves("za");
};

struct S_Drop_PreservesZA : S_PreservesZA {
// expected-cpp-error@+2 {{virtual function 'memberfn' has different attributes ('void ()') than the function it overrides (which has 'void () __arm_preserves("za")')}}
// expected-cpp-note@-5 {{overridden virtual function is here}}
  void memberfn(void) override {}
};

struct S_NoPreservesZA {
  virtual void memberfn(void);
};

struct S_AddPreservesZA : S_NoPreservesZA {
// expected-cpp-error@+2 {{virtual function 'memberfn' has different attributes ('void () __arm_preserves("za")') than the function it overrides (which has 'void ()')}}
// expected-cpp-note@-5 {{overridden virtual function is here}}
  void memberfn(void) __arm_preserves("za") override {}
};


// Check that the attribute propagates through template instantiations.
template <typename Ty>
struct S3 {
  static constexpr int value = 0;
};

template <>
struct S3<void (*)()> {
  static constexpr int value = 1;
};

template <>
struct S3<void (* __arm_streaming)()> {
  static constexpr int value = 2;
};

template <>
struct S3<void (* __arm_streaming_compatible)()> {
  static constexpr int value = 4;
};

template <>
struct S3<void (* __arm_inout("za"))()> {
  static constexpr int value = 8;
};

template <>
struct S3<void (* __arm_preserves("za"))()> {
  static constexpr int value = 16;
};

void normal_func(void) {}
void streaming_func(void) __arm_streaming {}
void streaming_compatible_func(void) __arm_streaming_compatible {}
void shared_za_func(void) __arm_inout("za") {}
void preserves_za_func(void) __arm_preserves("za") {}

static_assert(S3<decltype(+normal_func)>::value == 1, "why are we picking the wrong specialization?");
static_assert(S3<decltype(+streaming_func)>::value == 2, "why are we picking the wrong specialization?");
static_assert(S3<decltype(+streaming_compatible_func)>::value == 4, "why are we picking the wrong specialization?");
static_assert(S3<decltype(+shared_za_func)>::value == 8, "why are we picking the wrong specialization?");
static_assert(S3<decltype(+preserves_za_func)>::value == 16, "why are we picking the wrong specialization?");

// Also test the attribute is propagated with variadic templates
constexpr int eval_variadic_template() { return 0; }
template <typename T, typename... Other>
constexpr int eval_variadic_template(T f, Other... other) {
    return S3<decltype(f)>::value + eval_variadic_template(other...);
}
static_assert(eval_variadic_template(normal_func, streaming_func,
                                     streaming_compatible_func,
                                     shared_za_func, preserves_za_func) == 31,
              "attributes  not propagated properly in variadic template");

// Test that the attribute is propagated with template specialization.
template<typename T> int test_templated_f(T);
template<> constexpr int test_templated_f<void(*)(void)>(void(*)(void)) { return 1; }
template<> constexpr int test_templated_f<void(*)(void)__arm_streaming>(void(*)(void)__arm_streaming) { return 2; }
template<> constexpr int test_templated_f<void(*)(void)__arm_streaming_compatible>(void(*)(void)__arm_streaming_compatible) { return 4; }
template<> constexpr int test_templated_f<void(*)(void)__arm_inout("za")>(void(*)(void)__arm_inout("za")) { return 8; }
template<> constexpr int test_templated_f<void(*)(void)__arm_preserves("za")>(void(*)(void)__arm_preserves("za")) { return 16; }

static_assert(test_templated_f(&normal_func) == 1, "Instantiated to wrong function");
static_assert(test_templated_f(&streaming_func) == 2, "Instantiated to wrong function");
static_assert(test_templated_f(&streaming_compatible_func) == 4, "Instantiated to wrong function");
static_assert(test_templated_f(&shared_za_func) == 8, "Instantiated to wrong function");
static_assert(test_templated_f(&preserves_za_func) == 16, "Instantiated to wrong function");

// expected-cpp-error@+2 {{'__arm_streaming' only applies to function types; type here is 'int'}}
// expected-error@+1 {{'__arm_streaming' only applies to function types; type here is 'int'}}
int invalid_type_for_attribute __arm_streaming;

// Test overloads
constexpr int overload(void f(void)) { return 1; }
constexpr int overload(void f(void) __arm_streaming) { return 2; }
constexpr int overload(void f(void) __arm_streaming_compatible) { return 4; }
constexpr int overload(void f(void) __arm_inout("za")) { return 8; }
constexpr int overload(void f(void) __arm_preserves("za")) { return 16; }
static_assert(overload(&normal_func) == 1, "Overloaded to wrong function");
static_assert(overload(&streaming_func) == 2, "Overloaded to wrong function");
static_assert(overload(&streaming_compatible_func) == 4, "Overloaded to wrong function");
static_assert(overload(&shared_za_func) == 8, "Overloaded to wrong function");
static_assert(overload(&preserves_za_func) == 16, "Overloaded to wrong function");

// Test implicit instantiation
template <typename T> struct X {
  static void foo(T) __arm_streaming { }
};
constexpr int overload_int(void f(int)) { return 1; }
constexpr int overload_int(void f(int) __arm_streaming) { return 2; }
constexpr X<int> *ptr = 0;
static_assert(overload_int(ptr->foo) == 2, "Overloaded to the wrong function after implicit instantiation");

#endif // ifdef __cplusplus

// expected-cpp-error@+2 {{unknown state ''}}
// expected-error@+1 {{unknown state ''}}
__arm_new("") void invalid_arm_new_empty_string(void);
// expected-cpp-error@+2 {{expected string literal as argument of '__arm_new' attribute}}
// expected-error@+1 {{expected string literal as argument of '__arm_new' attribute}}
__arm_new(0) void invalid_arm_new_non_literal_string(void);
// expected-cpp-error@+2 {{unknown state 'unknownstate'}}
// expected-error@+1 {{unknown state 'unknownstate'}}
__arm_new("unknownstate") void invalid_arm_new_unknown_state(void);

// expected-cpp-error@+2 {{unknown state ''}}
// expected-error@+1 {{unknown state ''}}
void invalid_arm_in_empty_string(void) __arm_in("");
// expected-cpp-error@+2 {{expected string literal as argument of '__arm_in' attribute}}
// expected-error@+1 {{expected string literal as argument of '__arm_in' attribute}}
void invalid_arm_in_non_literal_string(void) __arm_in(0);
// expected-cpp-error@+2 {{unknown state 'unknownstate'}}
// expected-error@+1 {{unknown state 'unknownstate'}}
void invalid_arm_in_unknown_state(void) __arm_in("unknownstate");

void valid_state_attrs_in_in1(void) __arm_in("za");
void valid_state_attrs_in_in2(void) __arm_in("za", "za");
void valid_state_attrs_in_in3(void) __arm_in("zt0");
void valid_state_attrs_in_in4(void) __arm_in("zt0", "zt0");
void valid_state_attrs_in_in5(void) __arm_in("za", "zt0");
__arm_new("za") void valid_state_attrs_in_in6(void) __arm_in("zt0");
__arm_new("zt0") void valid_state_attrs_in_in7(void) __arm_in("za");

// expected-cpp-error@+2 {{missing state for '__arm_in'}}
// expected-error@+1 {{missing state for '__arm_in'}}
void invalid_state_attrs_no_arg1(void) __arm_in();
// expected-cpp-error@+2 {{missing state for '__arm_new'}}
// expected-error@+1 {{missing state for '__arm_new'}}
__arm_new() void invalid_state_attrs_no_arg2(void);

// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_in_out(void) __arm_in("za") __arm_out("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_in_inout(void) __arm_in("za") __arm_inout("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_in_preserves(void) __arm_in("za") __arm_preserves("za");

// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_out_in(void) __arm_out("za") __arm_in("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_out_inout(void) __arm_out("za") __arm_inout("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_out_preserves(void) __arm_out("za") __arm_preserves("za");

// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_inout_in(void) __arm_inout("za") __arm_in("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_inout_out(void) __arm_inout("za") __arm_out("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_inout_preserves(void) __arm_inout("za") __arm_preserves("za");

// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_preserves_in(void) __arm_preserves("za") __arm_in("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_preserves_out(void) __arm_preserves("za") __arm_out("za");
// expected-cpp-error@+2 {{conflicting attributes for state 'za'}}
// expected-error@+1 {{conflicting attributes for state 'za'}}
void conflicting_state_attrs_preserves_inout(void) __arm_preserves("za") __arm_inout("za");

// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_in_out_zt0(void) __arm_in("zt0") __arm_out("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_in_inout_zt0(void) __arm_in("zt0") __arm_inout("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_in_preserves_zt0(void) __arm_in("zt0") __arm_preserves("zt0");

// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_out_in_zt0(void) __arm_out("zt0") __arm_in("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_out_inout_zt0(void) __arm_out("zt0") __arm_inout("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_out_preserves_zt0(void) __arm_out("zt0") __arm_preserves("zt0");

// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_inout_in_zt0(void) __arm_inout("zt0") __arm_in("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_inout_out_zt0(void) __arm_inout("zt0") __arm_out("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_inout_preserves_zt0(void) __arm_inout("zt0") __arm_preserves("zt0");

// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_preserves_in_zt0(void) __arm_preserves("zt0") __arm_in("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_preserves_out_zt0(void) __arm_preserves("zt0") __arm_out("zt0");
// expected-cpp-error@+2 {{conflicting attributes for state 'zt0'}}
// expected-error@+1 {{conflicting attributes for state 'zt0'}}
void conflicting_state_attrs_preserves_inout_zt0(void) __arm_preserves("zt0") __arm_inout("zt0");

// Test that we get a diagnostic for unimplemented case.
void unimplemented_spill_fill_za(void (*share_zt0_only)(void) __arm_inout("zt0")) __arm_inout("za", "zt0") {
  // expected-cpp-error@+4 {{call to a function that shares state other than 'za' from a function that has live 'za' state requires a spill/fill of ZA, which is not yet implemented}}
  // expected-cpp-note@+3 {{add '__arm_preserves("za")' to the callee if it preserves ZA}}
  // expected-error@+2 {{call to a function that shares state other than 'za' from a function that has live 'za' state requires a spill/fill of ZA, which is not yet implemented}}
  // expected-note@+1 {{add '__arm_preserves("za")' to the callee if it preserves ZA}}
  share_zt0_only();
}

// expected-cpp-error@+2 {{streaming function cannot be multi-versioned}}
// expected-error@+1 {{streaming function cannot be multi-versioned}}
__attribute__((target_version("sme2")))
void cannot_work_version(void) __arm_streaming {}
// expected-cpp-error@+5 {{function declared 'void ()' was previously declared 'void () __arm_streaming', which has different SME function attributes}}
// expected-cpp-note@-2 {{previous declaration is here}}
// expected-error@+3 {{function declared 'void (void)' was previously declared 'void (void) __arm_streaming', which has different SME function attributes}}
// expected-note@-4 {{previous declaration is here}}
__attribute__((target_version("default")))
void cannot_work_version(void) {}


// expected-cpp-error@+2 {{streaming function cannot be multi-versioned}}
// expected-error@+1 {{streaming function cannot be multi-versioned}}
__attribute__((target_clones("sme2")))
void cannot_work_clones(void) __arm_streaming {}


__attribute__((target("sme2")))
void just_fine_streaming(void) __arm_streaming {}
__attribute__((target_version("sme2")))
void just_fine(void) { just_fine_streaming(); }
__attribute__((target_version("default")))
void just_fine(void) {}


__arm_locally_streaming
__attribute__((target_version("sme2")))
void just_fine_locally_streaming(void) {}
__attribute__((target_version("default")))
void just_fine_locally_streaming(void) {}


void fmv_caller() {
    cannot_work_version();
    cannot_work_clones();
    just_fine();
    just_fine_locally_streaming();
}
