// RUN: %clang_cc1 -fsyntax-only -fblocks -fcxx-exceptions -verify -Wfunction-effects %s
// RUN: %clang_cc1 -fsyntax-only -fblocks -verify -x c -std=c23 -Wfunction-effects %s

#if !__has_attribute(nonblocking)
#error "the 'nonblocking' attribute is not available"
#endif

// --- ATTRIBUTE SYNTAX: SUBJECTS ---

int nl_var [[clang::nonblocking]]; // expected-warning {{'nonblocking' only applies to function types; type here is 'int'}}
struct nl_struct {} [[clang::nonblocking]]; // expected-warning {{attribute 'nonblocking' is ignored, place it after "struct" to apply attribute to type declaration}}
struct [[clang::nonblocking]] nl_struct2 {}; // expected-error {{'nonblocking' attribute cannot be applied to a declaration}}

// Positive case
typedef void (*fo)() [[clang::nonblocking]];
void (*read_me_and_weep(
  int val, void (*func)(int) [[clang::nonblocking]])
  [[clang::nonblocking]]) (int)
  [[clang::nonblocking]];

// --- ATTRIBUTE SYNTAX: ARGUMENT COUNT ---
void nargs_1() [[clang::nonblocking(1, 2)]];  // expected-error {{'nonblocking' attribute takes no more than 1 argument}}
void nargs_2() [[clang::nonallocating(1, 2)]]; // expected-error {{'nonallocating' attribute takes no more than 1 argument}}
void nargs_3() [[clang::blocking(1)]]; // expected-error {{'blocking' attribute takes no arguments}}
void nargs_4() [[clang::allocating(1)]]; // expected-error {{'allocating' attribute takes no arguments}}

// --- ATTRIBUTE SYNTAX: COMBINATIONS ---
// Check invalid combinations of nonblocking/nonallocating attributes

void nl_true_false_1() [[clang::nonblocking(true)]] [[clang::blocking]]; // expected-error {{'blocking' and 'nonblocking' attributes are not compatible}}
void nl_true_false_2() [[clang::blocking]] [[clang::nonblocking(true)]]; // expected-error {{'nonblocking' and 'blocking' attributes are not compatible}}

void nl_true_false_3() [[clang::nonblocking, clang::blocking]]; // expected-error {{'blocking' and 'nonblocking' attributes are not compatible}}
void nl_true_false_4() [[clang::blocking, clang::nonblocking]]; // expected-error {{'nonblocking' and 'blocking' attributes are not compatible}}

void na_true_false_1() [[clang::nonallocating(true)]] [[clang::allocating]]; // expected-error {{'allocating' and 'nonallocating' attributes are not compatible}}
void na_true_false_2() [[clang::allocating]] [[clang::nonallocating(true)]]; // expected-error {{'nonallocating' and 'allocating' attributes are not compatible}}

void na_true_false_3() [[clang::nonallocating, clang::allocating]]; // expected-error {{'allocating' and 'nonallocating' attributes are not compatible}}
void na_true_false_4() [[clang::allocating, clang::nonallocating]]; // expected-error {{'nonallocating' and 'allocating' attributes are not compatible}}

void nl_true_na_true_1() [[clang::nonblocking]] [[clang::nonallocating]];
void nl_true_na_true_2() [[clang::nonallocating]] [[clang::nonblocking]];

void nl_true_na_false_1() [[clang::nonblocking]] [[clang::allocating]]; // expected-error {{'allocating' and 'nonblocking' attributes are not compatible}}
void nl_true_na_false_2() [[clang::allocating]] [[clang::nonblocking]]; // expected-error {{'nonblocking' and 'allocating' attributes are not compatible}}

void nl_false_na_true_1() [[clang::blocking]] [[clang::nonallocating]];
void nl_false_na_true_2() [[clang::nonallocating]] [[clang::blocking]];

void nl_false_na_false_1() [[clang::blocking]] [[clang::allocating]];
void nl_false_na_false_2() [[clang::allocating]] [[clang::blocking]];

// --- TYPE CONVERSIONS ---

void unannotated();
void nonblocking() [[clang::nonblocking]];
void nonallocating() [[clang::nonallocating]];
void type_conversions()
{
  // It's fine to remove a performance constraint.
  void (*fp_plain)();

  fp_plain = nullptr;
  fp_plain = unannotated;
  fp_plain = nonblocking;
  fp_plain = nonallocating;

  // Adding/spoofing nonblocking is unsafe.
  void (*fp_nonblocking)() [[clang::nonblocking]];
  fp_nonblocking = nullptr;
  fp_nonblocking = nonblocking;
  fp_nonblocking = unannotated; // expected-warning {{attribute 'nonblocking' should not be added via type conversion}}
  fp_nonblocking = nonallocating; // expected-warning {{attribute 'nonblocking' should not be added via type conversion}}

  // Adding/spoofing nonallocating is unsafe.
  void (*fp_nonallocating)() [[clang::nonallocating]];
  fp_nonallocating = nullptr;
  fp_nonallocating = nonallocating;
  fp_nonallocating = nonblocking; // no warning because nonblocking includes nonallocating
  fp_nonallocating = unannotated; // expected-warning {{attribute 'nonallocating' should not be added via type conversion}}
}

#ifdef __cplusplus
struct PTMF {
  void unannotated();
  void nonblocking() [[clang::nonblocking]];
  void nonallocating() [[clang::nonallocating]];
};

void type_conversions_ptmf()
{
  // It's fine to remove a performance constraint.
  void (PTMF::*ptmf_plain)() = nullptr;

  ptmf_plain = &PTMF::unannotated;
  ptmf_plain = &PTMF::nonblocking;
  ptmf_plain = &PTMF::nonallocating;

  // Adding/spoofing nonblocking is unsafe.
  void (PTMF::*fp_nonblocking)() [[clang::nonblocking]] = nullptr;
  fp_nonblocking = &PTMF::nonblocking;
  fp_nonblocking = &PTMF::unannotated; // expected-warning {{attribute 'nonblocking' should not be added via type conversion}}
  fp_nonblocking = &PTMF::nonallocating; // expected-warning {{attribute 'nonblocking' should not be added via type conversion}}

  // Adding/spoofing nonallocating is unsafe.
  void (PTMF::*fp_nonallocating)() [[clang::nonallocating]] = nullptr;
  fp_nonallocating = &PTMF::nonallocating;
  fp_nonallocating = &PTMF::nonblocking; // no warning because nonblocking includes nonallocating fp_nonallocating = unannotated;
  fp_nonallocating = &PTMF::unannotated; // expected-warning {{attribute 'nonallocating' should not be added via type conversion}}
}

// There was a bug: noexcept and nonblocking could be individually removed in conversion, but not both  
void type_conversions_2()
{
  auto receives_fp = [](void (*fp)()) {
  };
  
  auto ne = +[]() noexcept {};
  auto nl = +[]() [[clang::nonblocking]] {};
  auto nl_ne = +[]() noexcept [[clang::nonblocking]] {};
  
  receives_fp(ne);
  receives_fp(nl);
  receives_fp(nl_ne);
}
#endif

// --- VIRTUAL METHODS ---
// Attributes propagate to overridden methods, so no diagnostics except for conflicts.
// Check this in the syntax tests too.
#ifdef __cplusplus
struct Base {
  virtual void f1();
  virtual void nonblocking() noexcept [[clang::nonblocking]];
  virtual void nonallocating() noexcept [[clang::nonallocating]];
  virtual void f2() [[clang::nonallocating]]; // expected-note {{previous declaration is here}}
};

struct Derived : public Base {
  void f1() [[clang::nonblocking]] override;
  void nonblocking() noexcept override;
  void nonallocating() noexcept override;
  void f2() [[clang::allocating]] override; // expected-warning {{effects conflict when merging declarations; kept 'allocating', discarded 'nonallocating'}}
};
#endif // __cplusplus

// --- REDECLARATIONS ---

void f2();
void f2() [[clang::nonblocking]]; // expected-note {{previous declaration is here}}
void f2(); // expected-warning {{attribute 'nonblocking' on function does not match previous declaration}}
// Note: we verify that the attribute is actually seen during the constraints tests.

void f3() [[clang::blocking]]; // expected-note {{previous declaration is here}}
void f3() [[clang::nonblocking]]; // expected-warning {{effects conflict when merging declarations; kept 'blocking', discarded 'nonblocking'}}

// --- OVERLOADS ---
#ifdef __cplusplus
struct S {
  void foo(); // expected-note {{previous declaration is here}}
  void foo() [[clang::nonblocking]]; // expected-error {{class member cannot be redeclared}}
};
#endif // __cplusplus

// --- COMPUTED NONBLOCKING ---
void f4() [[clang::nonblocking(__builtin_memset)]] {} // expected-error {{nonblocking attribute requires an integer constant}}

#ifdef __cplusplus
// Unexpanded parameter pack
template <bool ...val>
void f5() [[clang::nonblocking(val /* NO ... here */)]] {} // expected-error {{expression contains unexpanded parameter pack 'val'}}

void f6() { f5<true, false>(); }

template <bool B>
void ambiguous() [[clang::nonblocking(B)]] [[clang::blocking]]; // expected-note {{candidate template ignored: substitution failure [with B = true]: 'blocking' and 'nonblocking' attributes are not compatible}}

void f7() {
  ambiguous<true>(); // expected-error {{no matching function for call to 'ambiguous'}}
  ambiguous<false>();
}
#endif // __cplusplus
