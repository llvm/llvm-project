// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify -std=c++11 -fms-extensions -Wno-delete-incomplete -Wno-unused-value %s
// expected-no-diagnostics

#define P(e) static_assert(noexcept(e), "expected nothrow")
#define N(e) static_assert(!noexcept(e), "expected throw")
#define B(b, e) static_assert(b == noexcept(e), "expectation failed")

void simple() {
  P(0);
  P(0 + 0);
  int i;
  P(i);
  P(sizeof(0));
  P(static_cast<int>(0));
  N(throw 0);
  N((throw 0, 0));
}

void nospec();
void allspec() throw(...);
void intspec() throw(int);
void emptyspec() throw();
void nothrowattr() __attribute__((nothrow));
void noexcept_true() noexcept;
void noexcept_false() noexcept(false);

void call() {
  N(nospec());
  N(allspec());
  N(intspec());
  P(emptyspec());
  P(nothrowattr());
  P(noexcept_true());
  N(noexcept_false());
}

void (*pnospec)();
void (*pallspec)() throw(...);
void (*pintspec)() throw(int);
void (*pemptyspec)() throw();

typedef void (*funcptr)();
funcptr returnsptr() throw();

void callptr() {
  N(pnospec());
  N((*pnospec)());
  N(pallspec());
  N((*pallspec)());
  N(pintspec());
  N((*pintspec)());
  P(pemptyspec());
  P((*pemptyspec)());
  N(returnsptr()());
}

struct S1 {
  void nospec();
  void allspec() throw(...);
  void intspec() throw(int);
  void emptyspec() throw();
};

void callmem() {
  S1 s;
  N(s.nospec());
  N(s.allspec());
  N(s.intspec());
  P(s.emptyspec());
}

void (S1::*mpnospec)();
void (S1::*mpallspec)() throw(...);
void (S1::*mpintspec)() throw(int);
void (S1::*mpemptyspec)() throw();

void callmemptr() {
  S1 s;
  N((s.*mpnospec)());
  N((s.*mpallspec)());
  N((s.*mpintspec)());
  P((s.*mpemptyspec)());
}

struct S2 {
  S2();
  S2(int, int) throw();
  void operator +();
  void operator -() throw();
  void operator +(int);
  void operator -(int) throw();
  operator int();
  operator float() throw();
};

void *operator new(__typeof__(sizeof(int)) sz, int) throw();

struct IncompleteStruct;

struct Bad1 {
  ~Bad1() throw(int);
};
struct Bad2 {
  void operator delete(void*) throw(int);
};

typedef int X;

void implicits() {
  N(new int);
  P(new (0) int);
  P(delete (int*)0);
  P(delete (IncompleteStruct*)0);
  N(delete (Bad1*)0);
  N(delete (Bad2*)0);
  N(S2());
  P(S2(0, 0));
  S2 s;
  N(+s);
  P(-s);
  N(s + 0);
  P(s - 0);
  N(static_cast<int>(s));
  P(static_cast<float>(s));
  N(Bad1());
  P(X().~X());
}

struct V {
  virtual ~V() throw();
};
struct D : V {};

void dyncast() {
  V *pv = 0;
  D *pd = 0;
  P(dynamic_cast<V&>(*pd));
  P(dynamic_cast<V*>(pd));
  N(dynamic_cast<D&>(*pv));
  P(dynamic_cast<D*>(pv));
}

namespace std {
  struct type_info {};
}

void idtype() {
  P(typeid(V));
  P(typeid((V*)0));
  P(typeid(*(S1*)0));
  N(typeid(*(V*)0));
}

void uneval() {
  P(sizeof(typeid(*(V*)0)));
  P(typeid(typeid(*(V*)0)));
}

struct G1 {};
struct G2 { int i; };
struct G3 { S2 s; };

void gencon() {
  P(G1());
  P(G2());
  N(G3());
}

template <class T> void f(T&&) noexcept;
template <typename T, bool b>
void late() {
  B(b, typeid(*(T*)0));
  B(b, T(1));
  B(b, static_cast<T>(S2(0, 0)));
  B(b, S1() + T());
  P(f(T()));
  P(new (0) T);
  P(delete (T*)0);
}
struct S3 {
  virtual ~S3() throw();
  S3() throw();
  explicit S3(int);
  S3(const S2&);
};
template <class T> T&& f2() noexcept;
template <typename T>
void late2() {
  P(dynamic_cast<S3&>(f2<T&>()));
}
void operator +(const S1&, float) throw();
void operator +(const S1&, const S3&);
void tlate() {
  late<float, true>();
  late<S3, false>();
  late2<S3>();
}
