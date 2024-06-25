// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s -fexceptions -fcxx-exceptions -Wno-unevaluated-expression
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s -fexceptions -fcxx-exceptions -Wno-unevaluated-expression -fexperimental-new-constant-interpreter

void f(); // expected-note {{possible target for call}}
void f(int); // expected-note {{possible target for call}}

void g() {
  bool b = noexcept(f); // expected-error {{reference to overloaded function could not be resolved; did you mean to call it with no arguments?}}
  bool b2 = noexcept(f(0));
}

struct S {
  void g(); // expected-note {{possible target for call}}
  void g(int); // expected-note {{possible target for call}}

  void h() {
    bool b = noexcept(this->g); // expected-error {{reference to non-static member function must be called; did you mean to call it with no arguments?}}
    bool b2 = noexcept(this->g(0));
  }
};

void stmt_expr() {
  static_assert(noexcept(({ 0; })));

  static_assert(!noexcept(({ throw 0; })));

  static_assert(noexcept(({
    try {
      throw 0;
    } catch (...) {
    }
    0;
  })));

  static_assert(!noexcept(({
    try {
      throw 0;
    } catch (...) {
      throw;
    }
    0;
  })));

  static_assert(!noexcept(({
    try {
      throw 0;
    } catch (int) {
    }
    0;
  })));

  static_assert(!noexcept(({
    if (false) throw 0;
  })));

  static_assert(noexcept(({
    if constexpr (false) throw 0;
  })));

  static_assert(!noexcept(({
    if constexpr (false) throw 0; else throw 1;
  })));

  static_assert(noexcept(({
    if constexpr (true) 0; else throw 1;
  })));
}

void vla(bool b) { // expected-note 5{{declared here}}
  static_assert(noexcept(static_cast<int(*)[true ? 41 : 42]>(0)), "");
  // FIXME: This can't actually throw, but we conservatively assume any VLA
  // type can throw for now.
  static_assert(!noexcept(static_cast<int(*)[b ? 41 : 42]>(0)), "");         // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                                                                expected-note {{function parameter 'b' with unknown value cannot be used in a constant expression}}
  static_assert(!noexcept(static_cast<int(*)[b ? throw : 42]>(0)), "");      // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                                                                expected-note {{function parameter 'b' with unknown value cannot be used in a constant expression}}
  static_assert(!noexcept(reinterpret_cast<int(*)[b ? throw : 42]>(0)), ""); // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                                                                expected-note {{function parameter 'b' with unknown value cannot be used in a constant expression}}
  static_assert(!noexcept((int(*)[b ? throw : 42])0), "");                   // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                                                                expected-note {{function parameter 'b' with unknown value cannot be used in a constant expression}}
  static_assert(!noexcept((int(*)[b ? throw : 42]){0}), "");                 // expected-warning {{variable length arrays in C++ are a Clang extension}} \
                                                                                expected-note {{function parameter 'b' with unknown value cannot be used in a constant expression}}
}

struct pr_44514 {
  // expected-error@+1{{value of type 'void' is not implicitly convertible to 'bool'}}
  void foo(void) const &noexcept(f());
};

namespace P1401 {
const int *ptr = nullptr;
void f() noexcept(sizeof(char[2])); // expected-error {{noexcept specifier argument evaluates to 2, which cannot be narrowed to type 'bool'}}
void g() noexcept(sizeof(char));
void h() noexcept(ptr);     // expected-error {{conversion from 'const int *' to 'bool' is not allowed in a converted constant expression}}
void i() noexcept(nullptr); // expected-error {{conversion from 'std::nullptr_t' to 'bool' is not allowed in a converted constant expression}}
void j() noexcept(0);
void k() noexcept(1);
void l() noexcept(2); // expected-error {{noexcept specifier argument evaluates to 2, which cannot be narrowed to type 'bool'}}
} // namespace P1401
