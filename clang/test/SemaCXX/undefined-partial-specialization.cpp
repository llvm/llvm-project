// RUN: %clang_cc1 -std=c++17 -verify %s
// RUN: %clang_cc1 -std=c++20 -verify %s

namespace GH61356 {

template <typename T, bool b>
class boo {void foo();};

template <typename T>
class boo<T, true>;

template<typename T>
void boo<T, true>::foo(){} // expected-error{{out-of-line definition of 'foo' from class 'GH61356::boo<T, true>' without definition}}

}
