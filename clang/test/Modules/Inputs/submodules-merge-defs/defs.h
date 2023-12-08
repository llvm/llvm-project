struct A { int a_member; };
inline int use_a(A a) { return a.a_member; }

class B {
  struct Inner1 {};
public:
  struct Inner2;
  struct Inner3;
  template<typename T> void f();
};
struct BFriend {
  friend class B::Inner3;
private:
  struct Inner3Base {};
};
// Check that lookup and access checks are performed in the right context.
struct B::Inner2 : Inner1 {};
struct B::Inner3 : BFriend::Inner3Base {};
template<typename T> void B::f() {}
template<> inline void B::f<int>() {}

// Check that base-specifiers are correctly disambiguated.
template<int N> struct C_Base { struct D { constexpr operator int() const { return 0; } }; };
const int C_Const = 0;
struct C1 : C_Base<C_Base<0>::D{}> {} extern c1;
struct C2 : C_Base<C_Const<0>::D{} extern c2;

typedef struct { int a; void f(); struct X; } D;
struct D::X { int dx; } extern dx;
inline int use_dx(D::X dx) { return dx.dx; }

template<typename T> int E(T t) { return t; }

template<typename T> struct F {
  int f();
  template<typename U> int g();
  static int n;
};
template<typename T> int F<T>::f() { return 0; }
template<typename T> template<typename U> int F<T>::g() { return 0; }
template<typename T> int F<T>::n = 0;
template<> inline int F<char>::f() { return 0; }
template<> template<typename U> int F<char>::g() { return 0; }
template<> struct F<void> { int h(); };
inline int F<void>::h() { return 0; }
template<typename T> struct F<T *> { int i(); };
template<typename T> int F<T*>::i() { return 0; }

namespace G {
  enum A { a, b, c, d, e };
  enum { f, g, h };
  typedef enum { i, j } k;
  typedef enum {} l;
}

template<typename T = int, int N = 3, template<typename> class K = F> int H(int a = 1);
template<typename T = int, int N = 3, template<typename> class K = F> using I = decltype(H<T, N, K>());
template<typename T = int, int N = 3, template<typename> class K = F> struct J {};

namespace NS {
  struct A {};
  template<typename T> struct B : A {};
  template<typename T> struct B<T*> : B<char> {};
  template<> struct B<int> : B<int*> {};
  inline void f() {}
}

namespace StaticInline {
  struct X {};
  static inline void f(X);
  static inline void g(X x) { f(x); }
}

namespace FriendDefArg {
  template<typename = int> struct A;
  template<int = 0> struct B;
  template<template<typename> class = A> struct C;
  template<typename = int, int = 0, template<typename> class = A> struct D {};
  template<typename U> struct Y {
    template<typename> friend struct A;
    template<int> friend struct B;
    template<template<typename> class> friend struct C;
    template<typename, int, template<typename> class> friend struct D;
  };
}

namespace SeparateInline {
  inline void f();
  void f() {}
  constexpr int g() { return 0; }
}

namespace TrailingAttributes {
  template<typename T> struct X {} __attribute__((aligned(8)));
}

namespace MergeFunctionTemplateSpecializations {
  template<typename T> T f();
  template<typename T> struct X {
    template<typename U> using Q = decltype(f<T>() + U());
  };
  using xiq = X<int>::Q<int>;
}

enum ScopedEnum : int;
enum ScopedEnum : int { a, b, c };

namespace RedeclDifferentDeclKind {
  struct X {};
  typedef X X;
  using RedeclDifferentDeclKind::X;
}

namespace Anon {
  struct X {
    union {
      int n;
    };
  };
}

namespace ClassTemplatePartialSpec {
  template<typename T> struct F;
  template<template<int> class A, int B> struct F<A<B>> {
    template<typename C> F();
  };
  template<template<int> class A, int B> template<typename C> F<A<B>>::F() {}

  template<typename A, int B> struct F<A[B]> {
    template<typename C> F();
  };
  template<typename A, int B> template<typename C> F<A[B]>::F() {}
}

struct MemberClassTemplate {
  template<typename T> struct A;
};
template<typename T> struct MemberClassTemplate::A {};
template<typename T> struct MemberClassTemplate::A<T*> {};
template<> struct MemberClassTemplate::A<int> {};
