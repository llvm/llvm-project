// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface B.cppm -o B.pcm
// RUN: %clang_cc1 -std=c++20 -fmodule-file=B=B.pcm -fsyntax-only C.cpp

//--- A.h
struct deduced {
  template<typename = void> auto operator()() const {}
};

//--- B.cppm
module;

#include "A.h"

export module B;

export template<class> void b() {
  deduced()();
}

//--- C.cpp
#include "A.h"
import B;
template void b<void>();
