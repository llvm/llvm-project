// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T, int N>
struct X0 {
  const char *f0(bool Cond) {
    return Cond? "honk" : N;
#if __cplusplus >= 201103L
// expected-error@-2 {{incompatible operand types ('const char *' and 'int')}}
#else
// expected-no-diagnostics
#endif
  }

  const char *f1(bool Cond) {
    return Cond? N : "honk";
#if __cplusplus >= 201103L
// expected-error@-2 {{incompatible operand types ('int' and 'const char *')}}
#endif
  }
  
  bool f2(const char *str) {
    return str == N;
#if __cplusplus >= 201103L
// expected-error@-2 {{comparison between pointer and integer ('const char *' and 'int')}}
#endif
  }
};

// PR4996
template<unsigned I> int f0() { 
  return __builtin_choose_expr(I, 0, 1); 
}

// PR5041
struct A { };

template <typename T> void f(T *t)
{
  (void)static_cast<void*>(static_cast<A*>(t));
}
