// RUN: %clang_cc1 -fsyntax-only -verify=expected,cxx23 -std=c++23 -Wpre-c++23-compat %s

// This test covers modifications made by P2448R2.

// Check that there is no error when a constexpr function that never produces a
// constant expression, but still an error if such function is called from
// constexpr context.
constexpr int F(int N) {  
  // cxx23-warning@-1 {{constexpr function that never produces a constant expression is incompatible with C++ standards before C++23}}
  double D = 2.0 / 0.0; // cxx23-note {{division by zero}} \
                        // expected-note {{division by zero}}
  return 1;
}

// No warning here since the function can produce a constant expression.
constexpr int F0(int N) {
  if (N == 0)
    double d2 = 2.0 / 0.0; // expected-note {{division by zero}}
  return 1;
}

template <typename T>
constexpr int FT(T N) {
  double D = 2.0 / 0.0; // expected-note {{division by zero}}
  return 1;
}

class NonLiteral {
// cxx23-note@-1 3{{'NonLiteral' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
public:
  NonLiteral() {} // cxx23-note 2{{declared here}}
  ~NonLiteral() {}
};

constexpr NonLiteral F1() {
// cxx23-warning@-1 {{constexpr function with non-literal return type 'NonLiteral' is incompatible with C++ standards before C++23}}
// cxx23-warning@-2 {{constexpr function that never produces a constant expression is incompatible with C++ standards before C++23}}
  return NonLiteral{};
// cxx23-note@-1 {{non-constexpr constructor 'NonLiteral' cannot be used in a constant expression}}
}

constexpr int F2(NonLiteral N) {
  // cxx23-warning@-1 {{constexpr function with 1st non-literal parameter type 'NonLiteral' is not compatible with C++ standards before C++23}}
  return 8;
}

class Derived : public NonLiteral {
  constexpr ~Derived() {}; // precxx20-error {{destructor cannot be declared constexpr}}
};

class Derived1 : public NonLiteral {
  constexpr Derived1() : NonLiteral () {}
  // cxx23-warning@-1{{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // cxx23-note@-2 {{non-constexpr constructor 'NonLiteral' cannot be used in a constant expression}}
};


struct X { // cxx23-note 2{{'X' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
  X(); // cxx23-note 3{{declared here}}
  X(const X&); // cxx23-note 2{{declared here}}
  X(X&&);
  X& operator=(X&); // cxx11-note 2{{not viable: 'this' argument has type 'const X', but method is not marked const}}
  X& operator=(X&& other); // cxx11-note 2{{not viable: 'this' argument has type 'const X', but method is not marked const}}
  bool operator==(X const&) const; // cxx23-note 2{{non-constexpr comparison function declared here}}
};

template <typename T>
struct Wrapper {
  constexpr Wrapper() = default;
  constexpr Wrapper(Wrapper const&) = default;
  constexpr Wrapper(T const& t) : t(t) { }
  constexpr Wrapper(Wrapper &&) = default;
  constexpr X get() const { return t; } // cxx23-warning {{constexpr function with non-literal return type 'X' is incompatible with C++ standards before C++23}}
  constexpr bool operator==(Wrapper const&) const = default; // cxx23-warning {{defaulted definition of equality comparison operator that is declared constexpr but invokes a non-constexpr comparison function is incompatible with C++ standards before C++23}}
  // precxx20-warning@-1 2{{defaulted comparison operators are a C++20 extension}}
private:
  T t; // cxx23-note {{non-constexpr comparison function would be used to compare member 't'}}
};

struct WrapperNonT {
  constexpr WrapperNonT() = default;
  constexpr WrapperNonT(WrapperNonT const&) = default;
  constexpr WrapperNonT(X const& t) : t(t) { } // cxx23-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // cxx23-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr WrapperNonT(WrapperNonT &&) = default;
  constexpr WrapperNonT& operator=(WrapperNonT &) = default;
  // cxx11-error@-1 {{an explicitly-defaulted copy assignment operator may not have 'const', 'constexpr' or 'volatile' qualifiers}}
  constexpr WrapperNonT& operator=(WrapperNonT&& other) = default;
  // cxx11-error@-1 {{an explicitly-defaulted move assignment operator may not have 'const', 'constexpr' or 'volatile' qualifiers}}
  constexpr X get() const { return t; } // cxx23-warning {{constexpr function with non-literal return type 'X' is incompatible with C++ standards before C++23}}
  // cxx23-warning@-1{{constexpr function that never produces a constant expression is incompatible with C++ standards before C++23}}
  // cxx23-note@-2 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr bool operator==(WrapperNonT const&) const = default; // cxx23-warning {{defaulted definition of equality comparison operator that is declared constexpr but invokes a non-constexpr comparison function is incompatible with C++ standards before C++23}}
  // precxx20-warning@-1 {{defaulted comparison operators are a C++20 extension}}
private:
  X t; // cxx23-note {{non-constexpr comparison function would be used to compare member 't'}}
};

struct NonDefaultMembers {
  constexpr NonDefaultMembers() {}; // cxx23-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // cxx23-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  // expected-note@-2 {{non-literal type 'X' cannot be used in a constant expression}}
  constexpr NonDefaultMembers(NonDefaultMembers const&) {}; // cxx23-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // cxx23-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr NonDefaultMembers(NonDefaultMembers &&) {}; // cxx23-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // cxx23-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr NonDefaultMembers& operator=(NonDefaultMembers &other) {this->t = other.t; return *this;}
  // cxx11-error@-1 {{no viable overloaded '='}}
  // cxx11-error@-2 {{binding reference of type 'NonDefaultMembers' to value of type 'const NonDefaultMembers' drops 'const' qualifier}}
  constexpr NonDefaultMembers& operator=(NonDefaultMembers&& other) {this->t = other.t; return *this;}
  // cxx11-error@-1 {{no viable overloaded '='}}
  // cxx11-error@-2 {{binding reference of type 'NonDefaultMembers' to value of type 'const NonDefaultMembers' drops 'const' qualifier}}
  constexpr bool operator==(NonDefaultMembers const& other) const {return this->t == other.t;}
  X t;
};

static int Glob = 0; // cxx23-note {{declared here}}
class C1 {
public:
  constexpr C1() : D(Glob) {}; // cxx23-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // cxx23-note@-1 {{read of non-const variable 'Glob' is not allowed in a constant expression}}
private:
  int D;
};

void test() {

  constexpr int A = F(3); // expected-error {{constexpr variable 'A' must be initialized by a constant expression}}
                          // expected-note@-1 {{in call}}
  F(3);
  constexpr int B = F0(0); // expected-error {{constexpr variable 'B' must be initialized by a constant expression}}
                           // expected-note@-1 {{in call}}
  F0(0);
  constexpr auto C = F1(); // cxx23-error {{constexpr variable cannot have non-literal type 'const NonLiteral'}}
  F1();
  NonLiteral L;
  constexpr auto D = F2(L); // cxx23-error {{constexpr variable 'D' must be initialized by a constant expression}}
                            // cxx23-note@-1 {{non-literal type 'NonLiteral' cannot be used in a constant expression}}

  constexpr auto E = FT(1); // expected-error {{constexpr variable 'E' must be initialized by a constant expression}}
                            // expected-note@-1 {{in call}}
  F2(L);

  Wrapper<X> x; // cxx23-note {{requested here}}
                // precxx20-note@-1 {{requested here}}
  WrapperNonT x1;
  NonDefaultMembers x2;

  // TODO these produce notes with an invalid source location.
  // static_assert((Wrapper<X>(), true));
  // static_assert((WrapperNonT(), true),""); 

  static_assert((NonDefaultMembers(), true),""); // expected-error{{expression is not an integral constant expression}} \
                                              // expected-note {{in call to}}
  constexpr bool FFF = (NonDefaultMembers() == NonDefaultMembers()); // expected-error{{must be initialized by a constant expression}} \
                                                                     // expected-note{{non-literal}}
}

struct A {
  A ();
  ~A();
};

template <class T>
struct opt
{
  union {
    char c;
    T data;
  };

  constexpr opt() {}

  constexpr ~opt()  {
   if (engaged)
     data.~T();
 }

  bool engaged = false;
};

consteval void foo() {
  opt<A> a;
}

void bar() { foo(); }
