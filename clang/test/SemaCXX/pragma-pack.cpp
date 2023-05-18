// RUN: %clang_cc1 -triple i686-apple-darwin9 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace rdar8745206 {

struct Base {
  int i;
};

#pragma pack(push, 1)
struct Sub : public Base {
  char c;
};
#pragma pack(pop)

int check[sizeof(Sub) == 5 ? 1 : -1];

}

namespace check2 {

struct Base {
  virtual ~Base();
  int x;
};

#pragma pack(push, 1)
struct Sub : virtual Base {
  char c;
};
#pragma pack(pop)

int check[sizeof(Sub) == 13 ? 1 : -1];

}

namespace llvm_support_endian {

template<typename, bool> struct X;

#pragma pack(push)
#pragma pack(1)
template<typename T> struct X<T, true> {
  T t;
};
#pragma pack(pop)

#pragma pack(push)
#pragma pack(2)
template<> struct X<long double, true> {
  long double c;
};
#pragma pack(pop)

int check1[__alignof(X<int, true>) == 1 ? 1 : -1];
int check2[__alignof(X<long double, true>) == 2 ? 1 : -1];

}
