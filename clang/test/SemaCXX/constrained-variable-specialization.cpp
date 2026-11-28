// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 -I%t %t/a.cpp -emit-module-interface -o %t/a.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t -I%t %t/b.cpp -verify -fsyntax-only

//--- header.hpp
template<typename> inline constexpr bool x = true;
template<class T> requires true inline constexpr bool x<T> = true;

//--- a.cpp
module;
#include "header.hpp"
export module a;
static_assert(x<void>);

//--- b.cpp
// expected-no-diagnostics
#include "header.hpp"
import a;

template<typename = void>
struct s;

template<typename T>
constexpr bool x<s<T>> = true;

static_assert(x<s<void>>);
