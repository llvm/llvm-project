// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics

template<class T> class Array { /* ... */ }; 
template<class T> void sort(Array<T>& v) { }

// instantiate sort(Array<int>&) - template-argument deduced
template void sort<>(Array<int>&);

template void sort(Array<long>&);

template<typename T, typename U> void f0(T, U*) { }

template void f0<int>(int, float*);
template void f0<>(double, float*);

template<typename T> struct hash { };
struct S {
  bool operator==(const S&) const { return false; }
};

template<typename T> struct Hash_map {
  void Method(const T& x) { h(x); }
  hash<T> h;
};

Hash_map<S> *x;
const Hash_map<S> *foo() {
  return x;
}

template<> struct hash<S> {
  int operator()(const S& k) const {
    return 0;
  }
};
