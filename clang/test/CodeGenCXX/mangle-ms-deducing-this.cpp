// RUN: %clang_cc1 %s -std=c++23 -triple=x86_64-linux -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -std=c++23 -triple=x86_64-win32 -emit-llvm -o - | FileCheck %s --check-prefix MS

namespace GH106660 {

template<auto> void f1();
template<auto&> void f2();
template<int (*)(int)> void f3();
template<int (&)(int)> void f4();

struct X {
    int f(this auto);
    int f(this int);
};

void test() {
    // CHECK: call void @_ZN8GH1066602f1ITnDaXadL_ZNHS_1X1fIiEEiT_EEEEvv
    // MS: call void @"??$f1@$1??$f@H@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f1<&X::f<int>>();
    // CHECK-NEXT: call void @_ZN8GH1066602f1ITnDaXadL_ZNHS_1X1fEiEEEEvv
    // MS-NEXT: call void @"??$f1@$1?f@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f1<static_cast<int (*)(int)>(&X::f)>();

    // CHECK-NEXT: call void @_ZN8GH1066602f2ITnRDaL_ZNHS_1X1fIiEEiT_EEEvv
    // MS-NEXT: call void @"??$f2@$1??$f@H@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f2<*&X::f<int>>();
    // CHECK-NEXT: call void @_ZN8GH1066602f2ITnRDaL_ZNHS_1X1fEiEEEvv
    // MS-NEXT: call void @"??$f2@$1?f@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f2<*static_cast<int (*)(int)>(&X::f)>();

    // CHECK-NEXT: call void @_ZN8GH1066602f3IXadL_ZNHS_1X1fIiEEiT_EEEEvv
    // MS-NEXT: call void @"??$f3@$1??$f@H@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f3<&X::f<int>>();
    // CHECK-NEXT: call void @_ZN8GH1066602f3IXadL_ZNHS_1X1fEiEEEEvv
    // MS-NEXT: call void @"??$f3@$1?f@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f3<static_cast<int (*)(int)>(&X::f)>();

    // CHECK-NEXT: call void @_ZN8GH1066602f4IL_ZNHS_1X1fIiEEiT_EEEvv
    // MS-NEXT: call void @"??$f4@$1??$f@H@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f4<*&X::f<int>>();
    // CHECK-NEXT: call void @_ZN8GH1066602f4IL_ZNHS_1X1fEiEEEvv
    // MS-NEXT: call void @"??$f4@$1?f@X@GH106660@@SAH_VH@Z@GH106660@@YAXXZ"
    f4<*static_cast<int (*)(int)>(&X::f)>();
}

} // namespace GH106660
