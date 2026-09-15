// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only
//
// RUN: %clang_cc1 -std=c++20 %t/M.cppm -emit-reduced-module-interface -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fprebuilt-module-path=%t -verify -fsyntax-only

//--- header.h
#ifndef HEADER_H
#define HEADER_H

struct X;
template <class> struct Y;
void f(struct X *, Y<int> *);

#endif

//--- M.cppm
module;
#include "header.h"

namespace N {
using ::X;
using ::Y;
}

export module M;

export namespace N {
using N::X;
using N::Y;
}

//--- use.cpp
// expected-no-diagnostics
import M;
using namespace N;

#include "header.h"
