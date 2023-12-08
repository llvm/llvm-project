// RUN: %clang_cc1 -fsyntax-only -verify %s

template<typename MetaFun, typename T>
struct bind_metafun {
  typedef typename MetaFun::template apply<T> type;
};

struct add_pointer {
  template<typename T>
  struct apply {
    typedef T* type;
  };
};

int i;
// FIXME: if we make the declarator below a pointer (e.g., with *ip),
// the error message isn't so good because we don't get the handy
// 'aka' telling us that we're dealing with an int**. Should we fix
// getDesugaredType to dig through pointers and such?
bind_metafun<add_pointer, int>::type::type ip = &i;
bind_metafun<add_pointer, float>::type::type fp = &i; // expected-error{{cannot initialize a variable of type 'bind_metafun<add_pointer, float>::type::type' (aka 'float *') with an rvalue of type 'int *'}}


template<typename T>
struct extract_type_type {
  typedef typename T::type::type t;
};

double d;
extract_type_type<bind_metafun<add_pointer, double> >::t dp = &d;
