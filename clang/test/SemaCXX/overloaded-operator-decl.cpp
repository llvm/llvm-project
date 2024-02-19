// RUN: %clang_cc1 -fsyntax-only -verify %s 
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s
struct X { 
  X();
  X(int); 
};

X operator+(X, X);
X operator-(X, X) { X x; return x; }

struct Y {
  Y operator-() const;
  void operator()(int x = 17) const;
  int operator[](int);

  static int operator+(Y, Y); // expected-error{{overloaded 'operator+' cannot be a static member function}}
};


void f(X x) {
  x = operator+(x, x);
}

X operator+(int, float); // expected-error{{overloaded 'operator+' must have at least one parameter of class or enumeration type}}

X operator*(X, X = 5); // expected-error{{parameter of overloaded 'operator*' cannot have a default argument}}

X operator/(X, X, ...); // expected-error{{overloaded 'operator/' cannot be variadic}}

X operator%(Y); // expected-error{{overloaded 'operator%' must be a binary operator (has 1 parameter)}}

void operator()(Y&, int, int); // expected-error{{overloaded 'operator()' must be a non-static member function}}

typedef int INT;
typedef float FLOAT;
Y& operator++(Y&);
Y operator++(Y&, INT);
X operator++(X&, FLOAT); // expected-error{{parameter of overloaded post-increment operator must have type 'int' (not 'FLOAT' (aka 'float'))}}

int operator+; // expected-error{{'operator+' cannot be the name of a variable or data member}}

namespace PR6238 {
  static struct {
    void operator()();
  } plus;
}

struct PR10839 {
  operator int; // expected-error{{'operator int' cannot be the name of a variable or data member}}
  int operator+; // expected-error{{'operator+' cannot be the name of a variable or data member}}
};

namespace PR14120 {
  struct A {
    static void operator()(int& i) { ++i; } // expected-warning{{is a C++23 extension}}
  };
  void f() {
    int i = 0;
    A()(i);
  }
}

namespace GH42535 {
class E {
  E& operator=(const E& rhs, ...); // expected-error{{overloaded 'operator=' cannot be variadic}}
  E& operator+=(const E& rhs, ...); // expected-error{{overloaded 'operator+=' cannot be variadic}}

};
void operator+(E, ...) {} // expected-error{{overloaded 'operator+' cannot be variadic}}
void operator-(E, ...) {} // expected-error{{overloaded 'operator-' cannot be variadic}}
void operator*(E, ...) {} // expected-error{{overloaded 'operator*' cannot be variadic}}
void operator/(E, ...) {} // expected-error{{overloaded 'operator/' must be a binary operator}}
void operator/(E, E, ...) {} // expected-error{{overloaded 'operator/' cannot be variadic}}
void operator%(E, ...) {} // expected-error{{overloaded 'operator%' must be a binary operator}}
void operator%(E, E, ...) {} // expected-error{{overloaded 'operator%' cannot be variadic}}
E& operator++(E&, ...); // expected-error{{overloaded 'operator++' cannot be variadic}}
E& operator--(E&, ...); // expected-error{{overloaded 'operator--' cannot be variadic}}
bool operator<(const E& lhs, ...); // expected-error{{overloaded 'operator<' must be a binary operator}}
bool operator<(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator>(const E& lhs, ...); // expected-error{{overloaded 'operator>' must be a binary operator}}
bool operator>(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator>=(const E& lhs, ...); // expected-error{{overloaded 'operator>=' must be a binary operator}}
bool operator>=(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator<=(const E& lhs, ...); // expected-error{{overloaded 'operator<=' must be a binary operator}}
bool operator<=(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator==(const E& lhs, ...); // expected-error{{overloaded 'operator==' must be a binary operator}}
bool operator==(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator!=(const E& lhs, ...); // expected-error{{overloaded 'operator!=' must be a binary operator}}
bool operator!=(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator&&(const E& lhs, ...); // expected-error{{overloaded 'operator&&' must be a binary operator}}
bool operator&&(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator||(const E& lhs, ...); // expected-error{{overloaded 'operator||' must be a binary operator}}
bool operator||(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator>>(const E& lhs, ...); // expected-error{{overloaded 'operator>>' must be a binary operator}}
bool operator>>(const E& lhs, const E& rhs, ...); // expected-error{{cannot be variadic}}
bool operator&(const E& lhs, ...); // expected-error{{cannot be variadic}}
#if __cplusplus >= 202002L
auto operator<=>(const E& lhs, ...);  // expected-error{{overloaded 'operator<=>' must be a binary operator}}
#endif
void d() {
  E() + E();
  E() - E();
  E() * E();
  E() / E();
  E() % E();
  ++E(); // expected-error{{cannot increment value of type 'E'}}
  --E(); // expected-error{{cannot decrement value of type 'E'}}
  E() < E();
  E() > E();
  E() <= E();
  E() >= E();
  E() == E();
  E() != E();
#if __cplusplus >= 202002L
  E() <=> E();
#endif
  E e;
  E e1 = e;
  e += e1;
  E() && E();
  E() || E();
  E() & E();
  E() >> E();
}
}
