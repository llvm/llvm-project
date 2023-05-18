// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -o - -emit-interface-stubs %s | FileCheck %s

// CHECK:      --- !ifs-v1
// CHECK-NEXT: IfsVersion: 3.0
// CHECK-NEXT: Target: x86_64-unknown-linux-gnu
// CHECK-NEXT: Symbols:
// CHECK-NEXT: - { Name: "a", Type: Object, Size: 4 }
// CHECK-NEXT: ...

template<typename T, T v> struct S9 {
    static constexpr T value = v;
};
template<typename T> struct S0 : public S9<bool, true> { };
template<typename T> constexpr bool CE2 = S0<T>::value;
int a = CE2<int>;
