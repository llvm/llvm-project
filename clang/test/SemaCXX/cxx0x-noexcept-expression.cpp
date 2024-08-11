// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s -fexceptions -fcxx-exceptions -Wno-unevaluated-expression
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++2a %s -fexceptions -fcxx-exceptions -Wno-unevaluated-expression -fexperimental-new-constant-interpreter

namespace std {
struct type_info;
}

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

namespace typeid_ {
template<bool NoexceptConstructor, bool NoexceptDestructor>
struct Polymorphic {
  Polymorphic() noexcept(NoexceptConstructor) {}
  virtual ~Polymorphic() noexcept(NoexceptDestructor) {}
};

static_assert(noexcept(typeid(Polymorphic<false, false>{})));  // Not evaluated (not glvalue)
static_assert(noexcept(typeid((Polymorphic<true, true>&&) Polymorphic<true, true>{})));
static_assert(!noexcept(typeid((Polymorphic<false, true>&&) Polymorphic<false, true>{})));
static_assert(!noexcept(typeid((Polymorphic<true, false>&&) Polymorphic<true, false>{})));
static_assert(!noexcept(typeid(*&(const Polymorphic<true, true>&) Polymorphic<true, true>{})));
static_assert(!noexcept(typeid(*&(const Polymorphic<false, true>&) Polymorphic<false, true>{})));
static_assert(!noexcept(typeid(*&(const Polymorphic<true, false>&) Polymorphic<true, false>{})));

template<bool B>
struct X {
  template<typename T> void f();
};
template<typename T>
void f1() {
  X<noexcept(typeid(*T{}))> dependent;
  // `dependent` should be type-dependent because the noexcept-expression should be value-dependent
  // (it is true if T is int*, false if T is Polymorphic<false, false>* for example)
  dependent.f<void>();  // This should need to be `.template f` to parse as a template
  // expected-error@-1 {{use 'template' keyword to treat 'f' as a dependent template name}}
}
template<typename... T>
void f2() {
  X<noexcept(typeid(*((static_cast<Polymorphic<false, false>*>(nullptr) && ... && T{}))))> dependent;
  // X<true> when T...[0] is a type with some operator&& which returns int*
  // X<false> when sizeof...(T) == 0
  dependent.f<void>();
  // expected-error@-1 {{use 'template' keyword to treat 'f' as a dependent template name}}
}
template<typename T>
void f3() {
  X<noexcept(typeid(*static_cast<T*>(nullptr)))> dependent;
  // X<true> when T is int, X<false> when T is Polymorphic<false, false>
  dependent.f<void>();
  // expected-error@-1 {{use 'template' keyword to treat 'f' as a dependent template name}}
}
template<typename T>
void f4() {
  X<noexcept(typeid(T))> not_dependent;
  not_dependent.non_existent();
  // expected-error@-1 {{no member named 'non_existent' in 'typeid_::X<true>'}}
}
template<typename T>
void f5() {
  X<noexcept(typeid(sizeof(sizeof(T))))> not_dependent;
  not_dependent.non_existent();
  // expected-error@-1 {{no member named 'non_existent' in 'typeid_::X<true>'}}
}
} // namespace typeid_

namespace GH97453 {

struct UnconstrainedCtor {
  int value_;

  template <typename T>
  constexpr UnconstrainedCtor(T value) noexcept(noexcept(value_ = value))
      : value_(static_cast<int>(value)) {}
};

UnconstrainedCtor U(42);

struct X {
  void ICE(int that) noexcept(noexcept([that]() {}));
};

} // namespace GH97453
