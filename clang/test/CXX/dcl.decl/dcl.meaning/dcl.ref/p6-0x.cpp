// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
// expected-no-diagnostics

template<typename T, typename U> 
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};
#define JOIN2(X,Y) X##Y
#define JOIN(X,Y) JOIN2(X,Y)
#define CHECK_EQUAL_TYPES(T1, T2) \
  int JOIN(array,__LINE__)[is_same<T1, T2>::value? 1 : -1]

int i; 
typedef int& LRI; 
typedef int&& RRI;

typedef LRI& r1; CHECK_EQUAL_TYPES(r1, int&);
typedef const LRI& r2; CHECK_EQUAL_TYPES(r2, int&);
typedef const LRI&& r3; CHECK_EQUAL_TYPES(r3, int&);

typedef RRI& r4; CHECK_EQUAL_TYPES(r4, int&);
typedef RRI&& r5; CHECK_EQUAL_TYPES(r5, int&&);
