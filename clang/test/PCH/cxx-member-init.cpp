// Test this without pch.
// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -DSOURCE -fsyntax-only -emit-llvm -o - %s

// Test with pch.
// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -std=c++11 -DHEADER -include-pch %t -fsyntax-only -emit-llvm -o - %s 

#ifdef HEADER
int n;
struct S {
  int *p = &m;
  int &m = n;
  S *that = this;
};
template<typename T> struct X { T t {0}; };
#endif

#ifdef SOURCE
S s;

struct E { explicit E(int); };
X<E> x;
#elif HEADER
#undef HEADER
#define SOURCE
#endif
