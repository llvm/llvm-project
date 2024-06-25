// RUN: %clang_cc1 -fsyntax-only -verify=expected -std=c++23 %s

// This test covers modifications made by P2448R2.

// Check that there is no error when a constexpr function that never produces a
// constant expression, but still an error if such function is called from
// constexpr context.
constexpr int F(int N) {
  double D = 2.0 / 0.0; // expected-note {{division by zero}}
  return 1;
}

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

class NonLiteral { // expected-note {{'NonLiteral' is not literal because it is not an aggregate and has no constexpr constructors}}
public:
  NonLiteral() {}
  ~NonLiteral() {}
};

constexpr NonLiteral F1() {
  return NonLiteral{};
}

constexpr int F2(NonLiteral N) {
  return 8;
}

class Derived : public NonLiteral {
  constexpr ~Derived() {};
};

class Derived1 : public NonLiteral {
  constexpr Derived1() : NonLiteral () {}
};


struct X {
  X();
  X(const X&);
  X(X&&);
  X& operator=(X&);
  X& operator=(X&& other);
  bool operator==(X const&) const;
};

template <typename T>
struct Wrapper {
  constexpr Wrapper() = default;
  constexpr Wrapper(Wrapper const&) = default;
  constexpr Wrapper(T const& t) : t(t) { }
  constexpr Wrapper(Wrapper &&) = default;
  constexpr X get() const { return t; }
  constexpr bool operator==(Wrapper const&) const = default;
  private:
  T t;
};

struct WrapperNonT {
  constexpr WrapperNonT() = default;
  constexpr WrapperNonT(WrapperNonT const&) = default;
  constexpr WrapperNonT(X const& t) : t(t) { }
  constexpr WrapperNonT(WrapperNonT &&) = default;
  constexpr WrapperNonT& operator=(WrapperNonT &) = default;
  constexpr WrapperNonT& operator=(WrapperNonT&& other) = default;
  constexpr X get() const { return t; }
  constexpr bool operator==(WrapperNonT const&) const = default;
  private:
  X t;
};

struct NonDefaultMembers {
  constexpr NonDefaultMembers() {}; // expected-note {{non-literal type 'X' cannot be used in a constant expression}}
  constexpr NonDefaultMembers(NonDefaultMembers const&) {};
  constexpr NonDefaultMembers(NonDefaultMembers &&) {};
  constexpr NonDefaultMembers& operator=(NonDefaultMembers &other) {this->t = other.t; return *this;}
  constexpr NonDefaultMembers& operator=(NonDefaultMembers&& other) {this->t = other.t; return *this;}
  constexpr bool operator==(NonDefaultMembers const& other) const {return this->t == other.t;}
  X t;
};

int Glob = 0;
class C1 {
public:
  constexpr C1() : D(Glob) {};
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

  Wrapper<X> x;
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
