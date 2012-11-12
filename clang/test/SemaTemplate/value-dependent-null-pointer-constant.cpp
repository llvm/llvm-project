// RUN: %clang_cc1 -fsyntax-only %s

template<typename T, int N>
struct X0 {
  const char *f0(bool Cond) {
    return Cond? "honk" : N;
  }

  const char *f1(bool Cond) {
    return Cond? N : "honk";
  }
  
  bool f2(const char *str) {
    return str == N;
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
