// RUN: %clang_cc1 -std=c++03 -fblocks -triple x86_64-windows-msvc -fms-extensions -fsyntax-only -fexceptions -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++11 -fblocks -triple x86_64-windows-msvc -fms-extensions -fsyntax-only -fexceptions -fcxx-exceptions -verify %s

// Basic usage should work.
int safe_div(int n, int d) {
  int r;
  __try {
    r = n / d;
  } __except(_exception_code() == 0xC0000094) {
    r = 0;
  }
  return r;
}

void might_crash();

// Diagnose obvious builtin mis-usage.
void bad_builtin_scope() {
  __try {
    might_crash();
  } __except(1) {
  }
  _exception_code(); // expected-error {{'_exception_code' only allowed in __except block or filter expression}}
  _exception_info(); // expected-error {{'_exception_info' only allowed in __except filter expression}}
}

// Diagnose obvious builtin misusage in a template.
template <void FN()>
void bad_builtin_scope_template() {
  __try {
    FN();
  } __except(1) {
  }
  _exception_code(); // expected-error {{'_exception_code' only allowed in __except block or filter expression}}
  _exception_info(); // expected-error {{'_exception_info' only allowed in __except filter expression}}
}
void instantiate_bad_scope_tmpl() {
  bad_builtin_scope_template<might_crash>();
}

#if __cplusplus < 201103L
template <typename T, T FN()>
T func_template() {
  return FN(); // expected-error 2{{builtin functions must be directly called}}
}
void inject_builtins() {
  func_template<void *, __exception_info>(); // expected-note {{instantiation of}}
  func_template<unsigned long, __exception_code>(); // expected-note {{instantiation of}}
}
#endif

void use_seh_after_cxx() {
  try { // expected-note {{conflicting 'try' here}}
    might_crash();
  } catch (int) {
  }
  __try { // expected-error {{cannot use C++ 'try' in the same function as SEH '__try'}}
    might_crash();
  } __except(1) {
  }
}

void use_cxx_after_seh() {
  __try { // expected-note {{conflicting '__try' here}}
    might_crash();
  } __except(1) {
  }
  try { // expected-error {{cannot use C++ 'try' in the same function as SEH '__try'}}
    might_crash();
  } catch (int) {
  }
}

#if __cplusplus >= 201103L
void use_seh_in_lambda() {
  ([]() {
    __try {
      might_crash();
    } __except(1) {
    }
  })();
  try {
    might_crash();
  } catch (int) {
  }
}
#endif

void use_seh_in_block() {
  void (^b)() = ^{
    __try { // expected-error {{cannot use SEH '__try' in blocks, captured regions, or Obj-C method decls}}
      might_crash();
    } __except(1) {
    }
  };
  try {
    b();
  } catch (int) {
  }
}

void (^use_seh_in_global_block)() = ^{
  __try { // expected-error {{cannot use SEH '__try' in blocks, captured regions, or Obj-C method decls}}
    might_crash();
  } __except(1) {
  }
};

void (^use_cxx_in_global_block)() = ^{
  try {
    might_crash();
  } catch(int) {
  }
};

template <class T> void dependent_filter() {
  __try {
    might_crash();
  } __except (T()) { // expected-error {{filter expression has non-integral type 'NotInteger'}}
  }
}

struct NotInteger { int x; };

void instantiate_dependent_filter() {
  dependent_filter<int>();
  dependent_filter<NotInteger>(); // expected-note {{requested here}}
}
