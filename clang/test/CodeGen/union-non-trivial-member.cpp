// RUN: %clang_cc1 --std=c++17 -emit-llvm %s -o - -triple x86_64-unknown-linux-gnu | FileCheck %s

struct non_trivial_constructor {
    constexpr non_trivial_constructor() : x(100) { }
    int x;
};

union UnionInt {
    int a{1000};
    non_trivial_constructor b;
};

union UnionNonTrivial {
    int a;
    non_trivial_constructor b{};
};

void f() {
    UnionInt u1;
    UnionNonTrivial u2;
}

// CHECK:      define dso_local void @_Z1fv()
// CHECK:        call void @_ZN8UnionIntC1Ev
// CHECK-NEXT:   call void @_ZN15UnionNonTrivialC1Ev

// CHECK:      define {{.*}}void @_ZN8UnionIntC1Ev
// CHECK:        call void @_ZN8UnionIntC2Ev

// CHECK:      define {{.*}}void @_ZN15UnionNonTrivialC1Ev
// CHECK:        call void @_ZN15UnionNonTrivialC2Ev

// CHECK:      define {{.*}}void @_ZN8UnionIntC2Ev
// CHECK:        store i32 1000

// CHECK:      define {{.*}}void @_ZN15UnionNonTrivialC2Ev
// CHECK:        call void @_ZN23non_trivial_constructorC1Ev
