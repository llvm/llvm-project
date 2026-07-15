// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/Base.cppm -emit-module-interface -o %t/Base.pcm
// RUN: %clang_cc1 -std=c++20 %t/A.cppm -emit-module-interface -o %t/A.pcm -fprebuilt-module-path=%t
// RUN: %clang_cc1 -std=c++20 %t/B.cppm -fsyntax-only -verify -fprebuilt-module-path=%t

//--- Base.cppm
export module Base;
export template <class T>
class Base {};

//--- A.cppm
export module A;
import Base;
struct S {};

export Base<S> a;

//--- B.cppm
// expected-no-diagnostics
export module B;

import A;
import Base;

struct S {};

export Base<S> b;
