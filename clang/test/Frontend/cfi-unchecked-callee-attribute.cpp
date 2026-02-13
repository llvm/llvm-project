// RUN: %clang_cc1 -Wall -Wno-unused -Wno-uninitialized -std=c++2b -verify %s

#define CFI_UNCHECKED_CALLEE __attribute__((cfi_unchecked_callee))

void unchecked(void) CFI_UNCHECKED_CALLEE {}
void checked(void) {}

void (*checked_ptr)(void) = unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
void (CFI_UNCHECKED_CALLEE *unchecked_ptr)(void) = unchecked;
void (CFI_UNCHECKED_CALLEE *from_normal)(void) = checked;
void (CFI_UNCHECKED_CALLEE *c_no_function_decay)(void) = &unchecked;
void (CFI_UNCHECKED_CALLEE __attribute__((noreturn)) *other_conflict)(void) = &checked; // expected-error{{cannot initialize a variable of type 'void (*)() __attribute__((noreturn)) __attribute__((cfi_unchecked_callee))' with an rvalue of type 'void (*)()'}}
void (CFI_UNCHECKED_CALLEE *arr[10])(void);
void (*cfi_elem)(void) = arr[1];  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
void (CFI_UNCHECKED_CALLEE *cfi_unchecked_elem)(void) = arr[1];
void (CFI_UNCHECKED_CALLEE &ref)(void) = unchecked;
void (CFI_UNCHECKED_CALLEE &ref2)(void) = *unchecked;
void (&ref_cfi_checked)(void) = unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void ()' discards 'cfi_unchecked_callee' attribute}}
void (&ref_cfi_checked2)(void) = *unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void ()' discards 'cfi_unchecked_callee' attribute}}

void (CFI_UNCHECKED_CALLEE *unchecked_from_deref)(void) = &*unchecked;
void (*checked_from_deref)(void) = &*unchecked;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}

typedef void (CFI_UNCHECKED_CALLEE unchecked_func_t)(void);
typedef void (checked_func_t)(void);
typedef void (CFI_UNCHECKED_CALLEE *unchecked_func_ptr_t)(void);
typedef void (*checked_func_ptr_t)(void);
checked_func_t *checked_func = unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
unchecked_func_t *unchecked_func = unchecked;

void CFI_UNCHECKED_CALLEE before_func(void);
CFI_UNCHECKED_CALLEE void before_return_type(void);
void (* CFI_UNCHECKED_CALLEE after_name)(void);

void UsageOnImproperTypes() {
  int CFI_UNCHECKED_CALLEE i;  // expected-warning{{'cfi_unchecked_callee' only applies to function types; type here is 'int'}}
}

/// Explicit casts suppress the warning.
void CheckCasts() {
  void (*should_warn)(void) = unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}

  void (*no_warn_c_style_cast)(void) = (void (*)(void))unchecked;
  void (*no_warn_static_cast)(void) = static_cast<void (*)(void)>(unchecked);
  void (*no_warn_reinterpret_cast)(void) = reinterpret_cast<void (*)(void)>(unchecked);
  unsigned long long ull = (unsigned long long)unchecked;

  struct A {};
  void (CFI_UNCHECKED_CALLEE A::*cfi_unchecked_member_ptr)(void);
  void (A::*member_ptr)(void) = cfi_unchecked_member_ptr;  // expected-warning{{implicit conversion from 'void (A::*)() __attribute__((cfi_unchecked_callee))' to 'void (A::*)()' discards 'cfi_unchecked_callee' attribute}}

  struct B {} CFI_UNCHECKED_CALLEE b;  // expected-warning{{'cfi_unchecked_callee' attribute only applies to functions and methods}}
  struct CFI_UNCHECKED_CALLEE C {} c;  // expected-warning{{'cfi_unchecked_callee' attribute only applies to functions and methods}}
  CFI_UNCHECKED_CALLEE struct D {} d;  // expected-warning{{'cfi_unchecked_callee' only applies to function types; type here is 'struct D'}}

  void *ptr2 = (void *)unchecked;
}

void CheckDifferentConstructions() {
  checked_func_t *checked_func(unchecked_func);  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
  new (checked_func_t *)(unchecked_func);  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
  struct S {
    checked_func_t *checked_func;

    S(unchecked_func_t *unchecked_func) : checked_func(unchecked_func) {}  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
  };

  checked_func_t *checked_func2{unchecked_func};  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
  checked_ptr = checked_func_ptr_t(unchecked);

  auto checked_auto = checked;
  auto unchecked_auto = unchecked;
  unchecked_ptr = checked_auto;
  unchecked_ptr = unchecked_auto;
  checked_ptr = checked_auto;
  checked_ptr = unchecked_auto;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
}

checked_func_t *returning_checked_func() {
  return unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
}

int checked_arg_func(checked_func_t *checked_func);
int invoke = checked_arg_func(unchecked);  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}

template <typename T>
struct S {
  S(T *ptr) {}
};
S<unchecked_func_t> s(checked);
S<unchecked_func_t> s2(unchecked);
S<checked_func_t> s3(checked);
S<checked_func_t> s4(unchecked);  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
S s5(checked);
S s6(unchecked);

template <typename T, typename U>
struct is_same {
  static constexpr bool value = false;
};
template <typename T>
struct is_same<T,T> {
  static constexpr bool value = true;
};

template <typename T>
struct ExpectingCFIUncheckedCallee {
  static_assert(is_same<T, unchecked_func_t>::value);
  ExpectingCFIUncheckedCallee(T *) {}
  ExpectingCFIUncheckedCallee() = default;
};
ExpectingCFIUncheckedCallee<unchecked_func_t> expecting;
ExpectingCFIUncheckedCallee expecting2(unchecked);

void no_args() __attribute__((cfi_unchecked_callee(10)));  // expected-error{{'cfi_unchecked_callee' attribute takes no arguments}}

void bracket_cfi_unchecked(void) [[clang::cfi_unchecked_callee]] {}

void BracketNotation() {
  checked_ptr = bracket_cfi_unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
}

void Comparisons() {
  /// Let's be able to compare checked and unchecked pointers without warnings.
  unchecked == checked_ptr;
  checked_ptr == unchecked;
  unchecked == unchecked_ptr;
  unchecked != checked_ptr;
  checked_ptr != unchecked;
  unchecked != unchecked_ptr;

  (void (*)(void))unchecked == checked_ptr;
  checked_ptr == (void (*)(void))unchecked;

  struct S {
    typedef void CB() CFI_UNCHECKED_CALLEE;
    constexpr bool operator==(const S &other) const {
      return cb == other.cb;
    }
    CB *cb;
  };
}

/// Type aliasing
typedef void (BaseType)(void);
using WithoutAttr = BaseType;
using WithAttr = __attribute__((cfi_unchecked_callee)) BaseType;

WithoutAttr *checked_func_alias = unchecked;  // expected-warning{{implicit conversion from 'void () __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
WithAttr *unchecked_func_allias = unchecked;
WithoutAttr *checked_func_alias2 = checked;
WithAttr *unchecked_func_alias2 = checked;

using MyType = WithAttr;  // expected-note{{previous definition is here}}
using MyType = WithoutAttr;  // expected-error{{type alias redefinition with different types ('WithoutAttr' (aka 'void ()') vs 'WithAttr' (aka 'void () __attribute__((cfi_unchecked_callee))'))}}

void MemberFunctionPointer() {
  struct A {
    void unchecked() CFI_UNCHECKED_CALLEE {}
    virtual void unchecked_virtual() CFI_UNCHECKED_CALLEE {}
    static void unchecked_static() CFI_UNCHECKED_CALLEE {}
    void unchecked_explicit_this(this A&) CFI_UNCHECKED_CALLEE {}
    int operator+=(int i) CFI_UNCHECKED_CALLEE { return i; }

    void checked() {}
    virtual void checked_virtual() {}
    static void checked_static() {}
    void checked_explicit_this(this A&) {}
    int operator-=(int i) { return i; }
  };

  void (CFI_UNCHECKED_CALLEE A::*unchecked)() = &A::unchecked;
  unchecked = &A::unchecked_virtual;
  void (CFI_UNCHECKED_CALLEE *unchecked_explicit_this)(A&) = &A::unchecked_explicit_this;
  void (CFI_UNCHECKED_CALLEE *unchecked_static)() = &A::unchecked_static;
  int (CFI_UNCHECKED_CALLEE A::*unchecked_overloaded)(int) = &A::operator+=;

  void (A::*checked)() = &A::unchecked;  // expected-warning{{implicit conversion from 'void (A::*)() __attribute__((cfi_unchecked_callee))' to 'void (A::*)()' discards 'cfi_unchecked_callee' attribute}}
  checked = &A::unchecked_virtual;  // expected-warning{{implicit conversion from 'void (A::*)() __attribute__((cfi_unchecked_callee))' to 'void (A::*)()' discards 'cfi_unchecked_callee' attribute}}
  void (*checked_explicit_this)(A&) = &A::unchecked_explicit_this;  // expected-warning{{implicit conversion from 'void (*)(A &) __attribute__((cfi_unchecked_callee))' to 'void (*)(A &)' discards 'cfi_unchecked_callee' attribute}}
  void (*checked_static)() = &A::unchecked_static;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
  int (A::*checked_overloaded)(int) = &A::operator+=;  // expected-warning{{implicit conversion from 'int (A::*)(int) __attribute__((cfi_unchecked_callee))' to 'int (A::*)(int)' discards 'cfi_unchecked_callee' attribute}}

  unchecked = &A::checked;
  unchecked = &A::checked_virtual;
  unchecked_explicit_this = &A::checked_explicit_this;
  unchecked_static = &A::checked_static;
  unchecked_overloaded = &A::operator-=;

  checked = &A::checked;
  checked = &A::checked_virtual;
  checked_explicit_this = &A::checked_explicit_this;
  checked_static = &A::checked_static;
  checked_overloaded = &A::operator-=;

  typedef void (CFI_UNCHECKED_CALLEE A::*WithAttr)();
  typedef void (CFI_UNCHECKED_CALLEE A::*WithoutAttr)();
  using WithoutAttr = decltype(unchecked);
}

void lambdas() {
  auto unchecked_lambda = [](void) CFI_UNCHECKED_CALLEE -> void {};
  auto checked_lambda = [](void) -> void {};
  void (CFI_UNCHECKED_CALLEE *unchecked_func)(void) = unchecked_lambda;
  unchecked_func = checked_lambda;
  void (*checked_func)(void) = unchecked_lambda;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
  checked_func = checked_lambda;

  auto capture_by_value = [unchecked_lambda, checked_lambda]() {
    void (CFI_UNCHECKED_CALLEE *unchecked_func)(void) = unchecked_lambda;
    unchecked_func = checked_lambda;
    void (*checked_func)(void) = unchecked_lambda;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
    checked_func = checked_lambda;
  };

  auto capture_by_ref = [&unchecked_lambda, &checked_lambda]() {
    void (CFI_UNCHECKED_CALLEE *unchecked_func)(void) = unchecked_lambda;
    unchecked_func = checked_lambda;
    void (*checked_func)(void) = unchecked_lambda;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
    checked_func = checked_lambda;
  };

  auto capture_all_by_value = [=]() {
    void (CFI_UNCHECKED_CALLEE *unchecked_func)(void) = unchecked_lambda;
    unchecked_func = checked_lambda;
    void (*checked_func)(void) = unchecked_lambda;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
    checked_func = checked_lambda;
  };

  auto capture_all_by_ref = [&]() {
    void (CFI_UNCHECKED_CALLEE *unchecked_func)(void) = unchecked_lambda;
    unchecked_func = checked_lambda;
    void (*checked_func)(void) = unchecked_lambda;  // expected-warning{{implicit conversion from 'void (*)() __attribute__((cfi_unchecked_callee))' to 'void (*)()' discards 'cfi_unchecked_callee' attribute}}
    checked_func = checked_lambda;
  };
}

CFI_UNCHECKED_CALLEE
void func(void);
void func(void) {}  // No warning expected.
