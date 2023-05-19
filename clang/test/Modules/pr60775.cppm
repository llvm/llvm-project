// https://github.com/llvm/llvm-project/issues/60775
//
// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/a.cppm -I%t -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 %t/b.cpp -fmodule-file=a=%t/a.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/c.cppm -I%t -emit-module-interface -o %t/c.pcm
// RUN: %clang_cc1 -std=c++20 %t/d.cppm -emit-module-interface -fmodule-file=c=%t/c.pcm -o %t/d.pcm
// RUN: %clang_cc1 -std=c++20 %t/e.cpp -fmodule-file=d=%t/d.pcm -fmodule-file=c=%t/c.pcm -verify -fsyntax-only
// RUN: %clang_cc1 -std=c++20 %t/f.cppm -emit-module-interface -fmodule-file=c=%t/c.pcm -o %t/f.pcm
// RUN: %clang_cc1 -std=c++20 %t/g.cpp -fmodule-file=f=%t/f.pcm -fmodule-file=c=%t/c.pcm  -verify -fsyntax-only

//--- initializer_list.h
namespace std {
  typedef decltype(sizeof(int)) size_t;
  template<typename T> struct initializer_list {
    initializer_list(const T *, size_t);
    T* begin();
    T* end();
  };
}

//--- a.cppm
module;
#include "initializer_list.h"
export module a;
export template<typename>
void a() {
	for (int x : {0}) {
	}
}

//--- b.cpp
// expected-no-diagnostics
import a;
void b() {
	a<int>();
}

//--- c.cppm
module;
#include "initializer_list.h"
export module c;
namespace std {
    export using std::initializer_list;
}

//--- d.cppm
export module d;
import c;
export template<typename>
void d() {
	for (int x : {0}) {
	}
}

//--- e.cpp
import d;
void e() {
    for (int x : {0}) { // expected-error {{cannot deduce type of initializer list because std::initializer_list was not found; include <initializer_list>}}
	}
}

template <typename>
void ee() {
    for (int x : {0}) { // expected-error {{cannot deduce type of initializer list because std::initializer_list was not found; include <initializer_list>}}
    }
}

void eee() {
    ee<int>();
    d<int>();
}

//--- f.cppm
export module f;
export import c;

//--- g.cpp
// expected-no-diagnostics
import f;
void g() {
    for (int x : {0}) {
	}
}

template <typename>
void gg() {
    for (int x : {0}) {
    }
}

void ggg() {
    gg<int>();
}
