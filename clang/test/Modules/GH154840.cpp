// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -fmodule-name=A -fno-cxx-modules -emit-module -fmodules -xc++ A.cppmap -o A.pcm
// RUN: %clang_cc1 -fmodule-name=B -fno-cxx-modules -emit-module -fmodules -xc++ B.cppmap -o B.pcm -fmodule-file=A.pcm
// RUN: %clang_cc1 -fmodule-name=C -fno-cxx-modules -emit-module -fmodules -xc++ C.cppmap -o C.pcm -fmodule-file=A.pcm
// RUN: %clang_cc1 -fmodule-name=D -fno-cxx-modules -emit-module -fmodules -xc++ D.cppmap -o D.pcm -fmodule-file=A.pcm
// RUN: %clang_cc1 -fmodule-name=E -fno-cxx-modules -emit-module -fmodules -xc++ E.cppmap -o E.pcm -fmodule-file=D.pcm -fmodule-file=B.pcm -fmodule-file=C.pcm
// RUN: %clang_cc1 -fno-cxx-modules -fmodules -fmodule-file=B.pcm -fmodule-file=E.pcm -emit-llvm -o /dev/null S.cpp

//--- A.h
namespace std {

template <class T> void zz(T);

template <class> struct vec {
  struct w {};
  struct xx {};

  vec(vec &) { init(); }
  constexpr vec &operator=(const vec &);
  template <class U> constexpr void pb(U);
  constexpr void init();

  w s;
};

template <class T> constexpr void vec<T>::init() {
  xx yy;
  zz(yy);
}

template <class T> constexpr vec<T> &vec<T>::operator=(const vec &) {
  pb(s);
  return *this;
}

template <class T> template <class U> constexpr void vec<T>::pb(U) { init(); }
} // namespace std

//--- A.cppmap
module "A" {
  header "A.h"
}

//--- X.h
#pragma clang module import A

namespace project {
  class thing : std::vec<thing> {};
} // namespace project

//--- B.h
#include "X.h"

//--- B.cppmap
module "B" {
  header "B.h"
}

//--- C.h
#include "X.h"

//--- C.cppmap
module "C" {
  header "C.h"
}

//--- D.h
#include "X.h"

//--- D.cppmap
module "D" {
  header "D.h"
}

//--- Y.h
#include "X.h"
struct other {
  other() : data(data) {}
  std::vec<project::thing> data;
};

//--- E.h
#include "Y.h"

//--- E.cppmap
module "E" {
  header "E.h"
}

//--- S.cpp
#pragma clang module import A
#pragma clang module import E
void func(std::vec<project::thing> *a, std::vec<project::thing> *b) { *a = *b; }
