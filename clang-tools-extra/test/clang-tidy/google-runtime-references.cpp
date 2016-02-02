// RUN: %check_clang_tidy %s google-runtime-references %t

int a;
int &b = a;
int *c;
void f1(int a);
void f2(int *b);
void f3(const int &c);
void f4(int const &d);

// Don't warn on implicit operator= in c++11 mode.
class A {
  virtual void f() {}
};
// Don't warn on rvalue-references.
struct A2 {
  A2(A2&&) = default;
  void f(A2&&) {}
};

// Don't warn on iostream parameters.
namespace xxx {
class istream { };
class ostringstream { };
}
void g1(xxx::istream &istr);
void g1(xxx::ostringstream &istr);

void g1(int &a);
// CHECK-MESSAGES: [[@LINE-1]]:14: warning: non-const reference parameter 'a', make it const or use a pointer [google-runtime-references]

struct s {};
void g2(int a, int b, s c, s &d);
// CHECK-MESSAGES: [[@LINE-1]]:31: warning: non-const reference parameter 'd', {{.*}}

typedef int &ref;
void g3(ref a);
// CHECK-MESSAGES: [[@LINE-1]]:13: warning: non-const reference {{.*}}

void g4(int &a, int &b, int &);
// CHECK-MESSAGES: [[@LINE-1]]:14: warning: non-const reference parameter 'a', {{.*}}
// CHECK-MESSAGES: [[@LINE-2]]:22: warning: non-const reference parameter 'b', {{.*}}
// CHECK-MESSAGES: [[@LINE-3]]:30: warning: non-const reference parameter '', {{.*}}

class B {
  B(B& a) {}
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: non-const reference {{.*}}
  virtual void f(int &a) {}
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: non-const reference {{.*}}
  void g(int &b);
// CHECK-MESSAGES: [[@LINE-1]]:15: warning: non-const reference {{.*}}

  // Don't warn on the parameter of stream extractors defined as members.
  B& operator>>(int& val) { return *this; }
};

// Only warn on the first declaration of each function to reduce duplicate
// warnings.
void B::g(int &b) {}

// Don't warn on the first parameter of stream inserters.
A& operator<<(A& s, int&) { return s; }
// CHECK-MESSAGES: [[@LINE-1]]:25: warning: non-const reference parameter '', {{.*}}

// Don't warn on either parameter of stream extractors. Both need to be
// non-const references by convention.
A& operator>>(A& input, int& val) { return input; }

// Don't warn on lambdas.
auto lambda = [] (int&) {};

// Don't warn on typedefs, as we'll warn on the function itself.
typedef int (*fp)(int &);

// Don't warn on function references.
typedef void F();
void g5(const F& func) {}
void g6(F& func) {}

template<typename T>
void g7(const T& t) {}

template<typename T>
void g8(T t) {}

void f5() {
  g5(f5);
  g6(f5);
  g7(f5);
  g7<F&>(f5);
  g8(f5);
  g8<F&>(f5);
}

// Don't warn on dependent types.
template<typename T>
void g9(T& t) {}
template<typename T>
void g10(T t) {}

void f6() {
  int i;
  float f;
  g9<int>(i);
  g9<const int>(i);
  g9<int&>(i);
  g10<int&>(i);
  g10<float&>(f);
}

// Warn only on the overridden methods from the base class, as the child class
// only implements the interface.
class C : public B {
  C();
  virtual void f(int &a) {}
};

// Don't warn on operator<< with streams-like interface.
A& operator<<(A& s, int) { return s; }

// Don't warn on swap().
void swap(C& c1, C& c2) {}

// Don't warn on standalone operator++, operator--, operator+=, operator-=,
// operator*=, etc. that all need non-const references to be functional.
A& operator++(A& a) { return a; }
A operator++(A& a, int) { return a; }
A& operator--(A& a) { return a; }
A operator--(A& a, int) { return a; }
A& operator+=(A& a, const A& b) { return a; }
A& operator-=(A& a, const A& b) { return a; }
A& operator*=(A& a, const A& b) { return a; }
A& operator/=(A& a, const A& b) { return a; }
A& operator%=(A& a, const A& b) { return a; }
A& operator<<=(A& a, const A& b) { return a; }
A& operator>>=(A& a, const A& b) { return a; }
A& operator|=(A& a, const A& b) { return a; }
A& operator^=(A& a, const A& b) { return a; }
A& operator&=(A& a, const A& b) { return a; }
