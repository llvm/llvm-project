#ifndef STRESS1_COMMON_H
#define STRESS1_COMMON_H

inline char function00(char x) { return x + x; }
inline short function00(short x) { return x + x; }
inline int function00(int x) { return x + x; }

namespace N01 { struct S00; }

namespace N00 {
struct S00 {
  char c;
  short s;
  int i;

  S00(char x) : c(x) {}
  S00(short x) : s(x) {}
  S00(int x) : i(x) {}

  char method00(char x) { return x + x; }
  short method00(short x) { return x + x; }
  int method00(int x) { return x + x; }

  operator char() { return c; }
  operator short() { return s; }
  operator int() { return i; }
};
struct S01 {};
struct S02 {};
template <typename T> struct S03 {
  struct S00 : N00::S00 {};
};
template <int I, template <typename> class U> struct S03<U<int>[I]>
    : U<int>::S00 {
  S03();
  S03(int);
  S03(short);
  S03(char);
  template <typename V = decltype(I)> S03(V);
};
template <> struct S03<S03<int>[42]> : S00 {};
}

namespace N01 {
struct S00 : N00::S00 {
  using N00::S00::S00;
};
struct S01 {};
struct S02 {};
}

using namespace N00;

template <int I, template <typename> class U> template <typename V> S03<U<int>[I]>::S03(V x) : S00(x) {}
template <int I, template <typename> class U> S03<U<int>[I]>::S03() : S00(I) {}
template <int I, template <typename> class U> S03<U<int>[I]>::S03(char x) : S00(x) {}
template <int I, template <typename> class U> S03<U<int>[I]>::S03(short x) : S00(x) {}
template <int I, template <typename> class U> S03<U<int>[I]>::S03(int x) : S00(x) {}

#pragma weak pragma_weak00
#pragma weak pragma_weak01
#pragma weak pragma_weak02
#pragma weak pragma_weak03
#pragma weak pragma_weak04
#pragma weak pragma_weak05

extern "C" int pragma_weak00();
extern "C" int pragma_weak01();
extern "C" int pragma_weak02();
extern "C" int pragma_weak03;
extern "C" int pragma_weak04;
extern "C" int pragma_weak05;

#endif
