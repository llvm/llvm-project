// RUN: rm -rf %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/M.cppm -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -emit-module-interface -fmodule-file=M=%t/M.pcm %t/B.cppm -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fmodule-file=M=%t/M.pcm -fmodule-file=B=%t/B.pcm %t/A.cppm
//
// Test again with reduced BMI
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/M.cppm -o %t/M.pcm
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface -fmodule-file=M=%t/M.pcm %t/B.cppm -o %t/B.pcm
// RUN: %clang_cc1 -std=c++20 -fsyntax-only -fmodule-file=M=%t/M.pcm -fmodule-file=B=%t/B.pcm %t/A.cppm

//--- decls.h
void f(int);

namespace N {
inline namespace I {
void f();
using ::f;
} // namespace I
} // namespace N

//--- M.cppm
module;
#include "decls.h"

export module M;

export namespace N { using N::f; }
export namespace N { using N::f; }

//--- B.cppm
export module B;

import M;

//--- A.cppm
module;
#include "decls.h"

export module A;

import B;

export void test() { N::f(); }
