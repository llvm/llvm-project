// RUN: %clang_cc1 -std=c++2d -verify=cxx29 -fsyntax-only -Wpre-c++2d-compat %s
// RUN: %clang_cc1 -std=c++11 -verify=ext -fsyntax-only -Wc++2d-extensions %s
// RUN: %clang_cc1 -std=c++98 -verify=ext -fsyntax-only -Wc++2d-extensions -Wno-c++11-extensions %s

template <class T> struct A {};
template <class T> struct B {};

template <template <class> class... TT>
struct S {
  // cxx29-warning@+2 {{pack indexing for template names is incompatible with C++ standards before C++2d}}
  // ext-warning@+1 {{pack indexing for template names is a C++2d extension}}
  typedef TT...[0]<int> a;
};

#if __cplusplus > 202302L
template <class T> concept Concept = true;

template <template <class> concept... CC>
struct Concepts {
  // cxx29-warning@+1 {{pack indexing for template names is incompatible with C++ standards before C++2d}}
  template <CC...[0] T>
  static void a();
};
#endif

template <class... T>
struct Types {
  // ext-warning@+1 {{pack indexing is a C++2c extension}}
  typedef T...[0] a;
};

template <class T> struct Deduce { Deduce(T); };

// ext-note@+1 {{template is declared here}}
template <template <class> class... TT>
void deduced_class_type() {
  // cxx29-warning@+3 {{pack indexing for template names is incompatible with C++ standards before C++2d}}
  // ext-warning@+2 {{pack indexing for template names is a C++2d extension}}
  // ext-error@+1 {{too few template arguments for template template parameter 'TT'}}
  TT...[0] x = 1;
  (void)x;
}

void use() {
  S<A, B> s;
  (void)s;
  Types<int, long> t;
  (void)t;
  deduced_class_type<Deduce>();
#if __cplusplus > 202302L
  Concepts<Concept> c;
  (void)c;
#endif
}
