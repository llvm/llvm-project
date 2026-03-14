// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: cd %t
//
// RUN: %clang_cc1 -std=c++20 -emit-module-interface %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
// RUN: %clang_cc1 -std=c++20 -emit-reduced-module-interface %t/A.cppm -o %t/A.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/Use.cpp -fsyntax-only -verify
//
//--- foo.h
template<typename T, typename U>
inline constexpr bool IsSame = false;

template<typename T>
inline constexpr bool IsSame<T, T> = true;

template <typename T>
class A {
public:
    A();
    ~A() noexcept(IsSame<T, T>);
};

//--- A.cppm
module;
#include "foo.h"
export module A;
export using ::A;

//--- Use.cpp
import A;
void bool_consume(bool b);
void use() {
    A<int> a{};
    bool_consume(IsSame); // expected-error {{use of undeclared identifier 'IsSame'}}
}
