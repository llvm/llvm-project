// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -std=c++11 -verify %s
// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -std=c++17 -verify %s

void f() const; // expected-error {{non-member function cannot have 'const' qualifier}}
void (*pf)() const; // expected-error {{pointer to function type cannot have 'const' qualifier}}
extern void (&rf)() const; // expected-error {{reference to function type cannot have 'const' qualifier}}

typedef void cfn() const;
cfn f2; // expected-error {{non-member function of type 'cfn' (aka 'void () const') cannot have 'const' qualifier}}

void decay1(void p() const); // expected-error {{non-member function cannot have 'const' qualifier}}
void decay2(cfn p); // expected-error {{non-member function of type 'cfn' (aka 'void () const') cannot have 'const' qualifier}}

class C {
  void f() const;
  cfn f2;
  static void f3() const; // expected-error {{static member function cannot have 'const' qualifier}}
  static cfn f4; // expected-error {{static member function of type 'cfn' (aka 'void () const') cannot have 'const' qualifier}}

  void m1() {
    x = 0;
  }

  void m2() const { // expected-note {{member function 'C::m2' is declared const here}}
    x = 0; // expected-error {{cannot assign to non-static data member within const member function 'm2'}}
  }

  int x;
};

void (C::*mpf)() const;
cfn C::*mpg;

// Don't crash!
void (PR14171)() const; // expected-error {{non-member function cannot have 'const' qualifier}}

// Test template instantiation of decayed array types.  Not really related to
// type quals.
template <typename T> void arrayDecay(const T a[]) { }
void instantiateArrayDecay() {
  int a[1];
  arrayDecay(a);
}

namespace GH79748 {
typedef decltype(sizeof(0)) size_t;
struct A {
  void* operator new(size_t bytes) const; //expected-error {{static member function cannot have 'const' qualifier}}
  void* operator new[](size_t bytes) const; //expected-error {{static member function cannot have 'const' qualifier}}

  void operator delete(void*) const; //expected-error {{static member function cannot have 'const' qualifier}}
  void operator delete[](void*) const; //expected-error {{static member function cannot have 'const' qualifier}}
};
struct B {
  void* operator new(size_t bytes) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}
  void* operator new[](size_t bytes) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}

  void operator delete(void*) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}
  void operator delete[](void*) volatile; //expected-error {{static member function cannot have 'volatile' qualifier}}
};
}

namespace GH27059 {
template<typename T> int f(T); // #GH27059-f
template<typename T, T> int g(); // #GH27059-g
int x = f<void () const>(nullptr);
// expected-error@-1 {{no matching function for call to 'f'}}
//   expected-note@#GH27059-f {{candidate template ignored: substitution failure [with T = void () const]: pointer to function type cannot have 'const' qualifier}}
int y = g<void () const, nullptr>();
// expected-error@-1 {{no matching function for call to 'g'}}
//   expected-note@#GH27059-g {{invalid explicitly-specified argument for 2nd template parameter}}

template<typename T> int ff(void p(T)); // #GH27059-ff
template<typename T, void(T)> int gg(); // #GH27059-gg
int xx = ff<void () const>(nullptr);
// expected-error@-1 {{no matching function for call to 'ff'}}
//   expected-note@#GH27059-ff {{candidate template ignored: substitution failure [with T = void () const]: pointer to function type cannot have 'const' qualifier}}
int yy = gg<void () const, nullptr>();
// expected-error@-1 {{no matching function for call to 'gg'}}
//   expected-note@#GH27059-gg {{invalid explicitly-specified argument for 2nd template parameter}}

template<typename T>
void catch_fn() {
  try {
  } catch (T) { // #GH27059-catch_fn
  }
}
template void catch_fn<void()>();
template void catch_fn<void() const>();
// expected-error@#GH27059-catch_fn {{pointer to function type 'void () const' cannot have 'const' qualifier}}
//   expected-note@-2 {{in instantiation of function template specialization 'GH27059::catch_fn<void () const>' requested here}}
template<typename T = int()const, T = nullptr> void f1() {}
template void f1();
// expected-error@-1 {{explicit instantiation of 'f1' does not refer to a function template, variable template, member function, member class, or static data member}}
//   expected-note@-3 {{candidate template ignored: substitution failure [with T = int () const]: pointer to function type cannot have 'const' qualifier}}

#if __cplusplus >= 201703L
template<typename T = void()const>
struct X { // expected-note {{candidate function template not viable: requires 1 argument, but 0 were provided}} \
              expected-note {{implicit deduction guide declared as 'template <typename T = void () const> X(X<T>) -> X<T>'}}
  X(T = 1); // expected-note {{candidate template ignored: substitution failure [with T = void () const]: pointer to function type 'void () const' cannot have 'const' qualifier}} \
               expected-note {{implicit deduction guide declared as 'template <typename T = void () const> X(T = <null expr>) -> X<T>'}}
};
void f2() {
  X{};
  // expected-error@-1 {{no viable constructor or deduction guide for deduction of template arguments of 'X'}}
}
#endif

}
