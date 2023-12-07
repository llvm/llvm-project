// RUN: %clang_cc1 -emit-pch -std=c++2a -o %t %s
// RUN: %clang_cc1 -std=c++2a -x ast -ast-print %t | FileCheck %s

// RUN: %clang_cc1 -emit-pch -std=c++2a -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -std=c++2a -x ast -ast-print %t | FileCheck %s

template<typename T>
concept C = true;

template<typename T, typename U>
concept C2 = true;

template<typename T>
bool f() {
  // CHECK: requires (T t) { t++; { t++ } noexcept -> C; { t++ } -> C2<int>; typename T::a; requires T::val; requires C<typename T::val> || (C<typename T::val> || C<T>); };
  return requires (T t) {
    t++;
    { t++ } noexcept -> C;
    { t++ } -> C2<int>;
    typename T::a;
    requires T::val;
    requires C<typename T::val> || (C<typename T::val> || C<T>);
  };
}
