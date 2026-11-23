// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t
//
// RUN: %clang_cc1 -std=c++20 %t/p1.cppm -emit-module-interface -o %t/p1.pcm
// RUN: %clang_cc1 -std=c++20 %t/p2.cppm -emit-module-interface -fmodule-file=m:p1=%t/p1.pcm -o %t/p2.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-module-interface -fmodule-file=m:p1=%t/p1.pcm -fmodule-file=m:p2=%t/p2.pcm -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=m=%t/m.pcm -fmodule-file=m:p1=%t/p1.pcm -fmodule-file=m:p2=%t/p2.pcm -fsyntax-only -verify

// RUN: %clang_cc1 -std=c++20 %t/p1.cppm -emit-reduced-module-interface -o %t/p1.pcm
// RUN: %clang_cc1 -std=c++20 %t/p2.cppm -emit-reduced-module-interface -fmodule-file=m:p1=%t/p1.pcm -o %t/p2.pcm
// RUN: %clang_cc1 -std=c++20 %t/m.cppm -emit-reduced-module-interface -fmodule-file=m:p1=%t/p1.pcm -fmodule-file=m:p2=%t/p2.pcm -o %t/m.pcm
// RUN: %clang_cc1 -std=c++20 %t/use.cpp -fmodule-file=m=%t/m.pcm -fmodule-file=m:p1=%t/p1.pcm -fmodule-file=m:p2=%t/p2.pcm -fsyntax-only -verify

//--- p1.cppm
export module m:p1;

namespace nn {
template<typename>
class Incognita;

export template<typename T>
class Variable
{
public:
  Incognita<T> foo() const { return {*this}; }
};
}

//--- p2.cppm
export module m:p2;

import :p1;

namespace nn {
template<typename T>
class Incognita
{
public:
  Incognita(const Variable<T> &) {}
};
}

//--- m.cppm
export module m;

export import :p1;
export import :p2;

//--- use.cpp
// expected-no-diagnostics
import m;

auto foo()
{
    nn::Variable<double> x;

    return x.foo();
}
