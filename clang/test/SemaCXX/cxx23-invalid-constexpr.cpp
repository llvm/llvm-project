// RUN: %clang_cc1 -fsyntax-only -verify -std=c++23 -Wpre-c++23-compat %s

// This test covers modifications made by P2448R2 in C++23 mode.

// Check that there is no error when a constexpr function that never produces a
// constant expression, but still an error if such function is called from
// constexpr context.
constexpr int F(int N) {  
  // expected-warning@-1 {{constexpr function that never produces a constant expression is incompatible with C++ standards before C++23}}
  double D = 2.0 / 0.0; // expected-note 2{{division by zero}}
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
// expected-note@-1 3{{'NonLiteral' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
public:
  NonLiteral() {} // expected-note 2{{declared here}}
  ~NonLiteral() {}
};

constexpr NonLiteral F1() {
// expected-warning@-1 {{constexpr function with non-literal return type 'NonLiteral' is incompatible with C++ standards before C++23}}
// expected-warning@-2 {{constexpr function that never produces a constant expression is incompatible with C++ standards before C++23}}
  return NonLiteral{};
// expected-note@-1 {{non-constexpr constructor 'NonLiteral' cannot be used in a constant expression}}
}

constexpr int F2(NonLiteral N) {
  // expected-warning@-1 {{constexpr function with 1st non-literal parameter type 'NonLiteral' is not compatible with C++ standards before C++23}}
  return 8;
}

class Derived : public NonLiteral { // expected-note {{declared here}}
  constexpr ~Derived() {};
  // expected-warning@-1{{constexpr destructor is incompatible with C++ standards before C++23 because base class 'NonLiteral' does not have a constexpr destructor}}

};

class Derived1 : public NonLiteral {
  constexpr Derived1() : NonLiteral () {}
  // expected-warning@-1{{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-2 {{non-constexpr constructor 'NonLiteral' cannot be used in a constant expression}}
};


struct X { // expected-note 2{{'X' is not literal because it is not an aggregate and has no constexpr constructors other than copy or move constructors}}
  X(); // expected-note 3{{declared here}}
  X(const X&); // expected-note 2{{declared here}}
  X(X&&);
  X& operator=(X&);
  X& operator=(X&& other);
  bool operator==(X const&) const; // expected-note 2{{non-constexpr comparison function declared here}}
};

template <typename T>
struct Wrapper {
  constexpr Wrapper() = default;
  constexpr Wrapper(Wrapper const&) = default;
  constexpr Wrapper(T const& t) : t(t) { }
  constexpr Wrapper(Wrapper &&) = default;
  constexpr X get() const { return t; } // expected-warning {{constexpr function with non-literal return type 'X' is incompatible with C++ standards before C++23}}
  constexpr bool operator==(Wrapper const&) const = default; // expected-warning {{defaulted definition of equality comparison operator that is declared constexpr but invokes a non-constexpr comparison function is incompatible with C++ standards before C++23}}
private:
  T t; // expected-note {{non-constexpr comparison function would be used to compare member 't'}}
};

struct WrapperNonT {
  constexpr WrapperNonT() = default; // expected-warning {{defaulted definition of default constructor that marked constexpr but never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-1 {{declared here}}
  constexpr WrapperNonT(WrapperNonT const&) = default; // expected-warning {{defaulted definition of copy constructor that marked constexpr but never produces a constant expression is incompatible with C++ standards before C++23}}
  constexpr WrapperNonT(X const& t) : t(t) { } // expected-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr WrapperNonT(WrapperNonT &&) = default; // expected-warning {{defaulted definition of move constructor that marked constexpr but never produces a constant expression is incompatible with C++ standards before C++23}}
  constexpr WrapperNonT& operator=(WrapperNonT &) = default; // expected-warning {{defaulted definition of copy assignment operator that marked constexpr but never produces a constant expression is incompatible with C++ standards before C++23}}
  constexpr WrapperNonT& operator=(WrapperNonT&& other) = default; // expected-warning {{defaulted definition of move assignment operator that marked constexpr but never produces a constant expression is incompatible with C++ standards before C++23}}
  constexpr X get() const { return t; } // expected-warning {{constexpr function with non-literal return type 'X' is incompatible with C++ standards before C++23}}
  // expected-warning@-1{{constexpr function that never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-2 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr bool operator==(WrapperNonT const&) const = default; // expected-warning {{defaulted definition of equality comparison operator that is declared constexpr but invokes a non-constexpr comparison function is incompatible with C++ standards before C++23}}
private:
  X t; // expected-note {{non-constexpr comparison function would be used to compare member 't'}}
};

struct NonDefaultMembers {
  constexpr NonDefaultMembers() {}; // expected-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  // expected-note@-2 {{non-literal type 'X' cannot be used in a constant expression}}
  constexpr NonDefaultMembers(NonDefaultMembers const&) {}; // expected-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr NonDefaultMembers(NonDefaultMembers &&) {}; // expected-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-1 {{non-constexpr constructor 'X' cannot be used in a constant expression}}
  constexpr NonDefaultMembers& operator=(NonDefaultMembers &other) {this->t = other.t; return *this;}
  constexpr NonDefaultMembers& operator=(NonDefaultMembers&& other) {this->t = other.t; return *this;}
  constexpr bool operator==(NonDefaultMembers const& other) const {return this->t == other.t;}
  X t;
};

static int Glob = 0; // expected-note {{declared here}}
class C1 {
public:
  constexpr C1() : D(Glob) {}; // expected-warning {{constexpr constructor that never produces a constant expression is incompatible with C++ standards before C++23}}
  // expected-note@-1 {{read of non-const variable 'Glob' is not allowed in a constant expression}}
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
  constexpr auto C = F1(); // expected-error {{constexpr variable cannot have non-literal type 'const NonLiteral'}}
  F1();
  NonLiteral L;
  constexpr auto D = F2(L); // expected-error {{constexpr variable 'D' must be initialized by a constant expression}}
                            // expected-note@-1 {{non-literal type 'NonLiteral' cannot be used in a constant expression}}

  constexpr auto E = FT(1); // expected-error {{constexpr variable 'E' must be initialized by a constant expression}}
                            // expected-note@-1 {{in call}}
  F2(L);

  Wrapper<X> x; // expected-note {{requested here}}
  WrapperNonT x1;
  NonDefaultMembers x2;

  // TODO produces note with an invalid source location
  // static_assert((Wrapper<X>(), true));
  
  static_assert((WrapperNonT(), true)); // expected-error{{expression is not an integral constant expression}}\
                                        // expected-note {{non-constexpr constructor 'WrapperNonT' cannot be used in a constant expression}}
  static_assert((NonDefaultMembers(), true)); // expected-error{{expression is not an integral constant expression}} \
                                              // expected-note {{in call to}}
  constexpr bool FFF = (NonDefaultMembers() == NonDefaultMembers()); // expected-error{{must be initialized by a constant expression}} \
                                                                     // expected-note{{non-literal}}

}
