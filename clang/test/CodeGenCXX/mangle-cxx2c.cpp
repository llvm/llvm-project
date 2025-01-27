// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-linux-gnu -std=c++2c | FileCheck %s

namespace GH112003 {

// CHECK-LABEL: define {{.*}} @_ZN8GH1120033fooILi0ETpTnDaJLi0ELi0EEEEDTsyT0_T_Ev
// CHECK-LABEL: define {{.*}} @_ZN8GH1120033fooILi1ETpTnDaJLi0ELi0EEEEDTsyT0_T_Ev
// CHECK-LABEL: define {{.*}} @_ZN8GH1120033fooILi0ETpTnDaJLl1EEEEDTsyT0_T_Ev
template <int I, auto...V>
decltype(V...[I]) foo() {return {};}

// CHECK-LABEL: define {{.*}} @_ZN8GH1120033barILi0EJilEEEDyT0_T_v
// CHECK-LABEL: define {{.*}} @_ZN8GH1120033barILi1EJilEEEDyT0_T_v
template <int I, typename...V>
V...[I] bar() {return {};}


template <int I, typename... T>
using First = T...[0];

// CHECK-LABEL: define {{.*}} @_ZN8GH1120033bazILi0EJiEEEvDy_SUBSTPACK_Li0E
// FIXME: handle indexing of partially substituted packs
template <int I, typename...V>
void baz(First<I, int, V...>){};


void fn() {
    foo<0, 0, 0>();
    foo<1, 0, 0>();
    foo<0, 1L>();
    bar<0, int, long>();
    bar<1, int, long>();
    baz<0, int>(0);
}
}
