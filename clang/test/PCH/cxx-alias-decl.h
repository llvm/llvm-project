// Header for PCH test cxx-alias-decl.cpp

struct S {};
template<typename U> struct T {
  template<typename V> using A = T<V>;
};

using A = int;
template<typename U> using B = S;
template<typename U> using C = T<U>;
template<typename U, typename V> using D = typename T<U>::template A<V>;
