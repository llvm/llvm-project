// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify %s -DORDER=1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify -std=c++98 %s -DORDER=1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify -std=c++11 %s -DORDER=1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify -std=c++17 %s -DORDER=1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify %s -DORDER=2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify -std=c++98 %s -DORDER=2
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify -std=c++11 %s -DORDER=2

// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify -std=c++17 %s -DORDER=2
// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -fmodules -fimplicit-module-maps -fexperimental-new-constant-interpreter -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-using-decls -verify -std=c++98 %s -DORDER=1

#if ORDER == 1
#include "a.h"
#include "b.h"
#else
#include "b.h"
#include "a.h"
#endif

struct Y {
  int value; // expected-note 0-1{{target of using}}
  typedef int type; // expected-note 0-1{{target of using}}
};

template<typename T> int Use() {
  int k = T().v + T().value; // expected-note 0-2{{instantiation of}}
  typedef typename T::type I;
  typedef typename T::t I;
  typedef int I;
  return k;
}

template<typename T> int UseAll() {
#if __cplusplus <= 199711L // C++11 does not allow access declarations
  return Use<C<T> >() + Use<D<T> >() + Use<E<T> >() + Use<F<T> >(); // expected-note 0-2{{instantiation of}}
#else
  return Use<C<T> >() + Use<D<T> >() + Use<F<T> >(); // expected-note 0-2{{instantiation of}}
#endif
}

template int UseAll<YA>();
template int UseAll<YB>();
template int UseAll<Y>();

#if __cplusplus >= 201702L
void use_g(Q q) {
  q.f(q); // expected-error {{ambiguous}}
#if ORDER == 1
  // expected-note@a.h:* {{candidate function}}
  // expected-note@a.h:* {{candidate function}}
#else
  // expected-note@b.h:* {{candidate function}}
  // expected-note@b.h:* {{candidate function}}
#endif
}
#endif

// Which of these two sets of diagnostics is chosen is not important. It's OK
// if this varies with ORDER, but it must be consistent across runs.
#if ORDER == 1
// Here, we're instantiating the definition from 'A' and merging the definition
// from 'B' into it.

#if __cplusplus <= 199711L // C++11 does not allow access declarations
// expected-error@b.h:* {{'E::value' from module 'B' is not present in definition of 'E<T>' in module 'A'}}
// expected-error@b.h:* {{'E::v' from module 'B' is not present in definition of 'E<T>' in module 'A'}}
#endif

// expected-error@b.h:* {{'F::type' from module 'B' is not present in definition of 'F<T>' in module 'A'}}
// expected-error@b.h:* {{'F::t' from module 'B' is not present in definition of 'F<T>' in module 'A'}}
// expected-error@b.h:* {{'F::value' from module 'B' is not present in definition of 'F<T>' in module 'A'}}
// expected-error@b.h:* {{'F::v' from module 'B' is not present in definition of 'F<T>' in module 'A'}}

// expected-note@a.h:* +{{does not match}}
#else
// Here, we're instantiating the definition from 'B' and merging the definition
// from 'A' into it.

// expected-error@a.h:* {{'D::type' from module 'A' is not present in definition of 'D<T>' in module 'B'}}
// expected-error@a.h:* {{'D::value' from module 'A' is not present in definition of 'D<T>' in module 'B'}}
// expected-error@b.h:* 2{{'typename' keyword used on a non-type}}
// expected-error@b.h:* 2{{dependent using declaration resolved to type without 'typename'}}

#if __cplusplus <= 199711L // C++11 does not allow access declarations
// expected-error@a.h:* {{'E::type' from module 'A' is not present in definition of 'E<T>' in module 'B'}}
// expected-error@a.h:* {{'E::t' from module 'A' is not present in definition of 'E<T>' in module 'B'}}
// expected-error@a.h:* {{'E::value' from module 'A' is not present in definition of 'E<T>' in module 'B'}}
// expected-error@a.h:* {{'E::v' from module 'A' is not present in definition of 'E<T>' in module 'B'}}
// expected-note@b.h:* 2{{definition has no member}}
#endif


// expected-error@a.h:* {{'F::type' from module 'A' is not present in definition of 'F<T>' in module 'B'}}
// expected-error@a.h:* {{'F::t' from module 'A' is not present in definition of 'F<T>' in module 'B'}}
// expected-error@a.h:* {{'F::value' from module 'A' is not present in definition of 'F<T>' in module 'B'}}
// expected-error@a.h:* {{'F::v' from module 'A' is not present in definition of 'F<T>' in module 'B'}}

// expected-note@b.h:* +{{does not match}}
// expected-note@b.h:* +{{target of using}}
#endif
