// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

namespace dr2338 {
namespace A {
enum E { Zero, One };
E a(int x) { return static_cast<E>(x); }
// CHECK-LABEL: define{{.*}} i32 @_ZN6dr23381A1aEi
// CHECK: ret i32 %0

E b(int x) { return (E)x; }
// CHECK-LABEL: define{{.*}} i32 @_ZN6dr23381A1bEi
// CHECK: ret i32 %0

} // namespace A
namespace B {
enum E : bool { Zero, One };
E a(int x) { return static_cast<E>(x); }
// CHECK-LABEL: define{{.*}} zeroext i1 @_ZN6dr23381B1aEi
// CHECK: ret i1 %tobool

E b(int x) { return (E)x; }
// CHECK-LABEL: define{{.*}} zeroext i1 @_ZN6dr23381B1bEi
// CHECK: ret i1 %tobool

} // namespace B
namespace C {
enum class E { Zero, One };
E a(int x) { return static_cast<E>(x); }
// CHECK-LABEL: define{{.*}} i32 @_ZN6dr23381C1aEi
// CHECK: ret i32 %0

E b(int x) { return (E)x; }
// CHECK-LABEL: define{{.*}} i32 @_ZN6dr23381C1bEi
// CHECK: ret i32 %0

} // namespace C
namespace D {
enum class E : bool { Zero, One };
E a(int x) { return static_cast<E>(x); }
// CHECK-LABEL: define{{.*}} zeroext i1 @_ZN6dr23381D1aEi
// CHECK: ret i1 %tobool

E b(int x) { return (E)x; }

// CHECK-LABEL: define{{.*}} zeroext i1 @_ZN6dr23381D1bEi
// CHECK: ret i1 %tobool

} // namespace D
} // namespace dr2338

namespace switchOnEnum {
enum class E : bool { Zero, One };
void func(E e) {
  switch (e) {
    case E::Zero...E::One:
      break;
  }
// CHECK-LABEL: define {{.*}}@_ZN12switchOnEnum4funcENS_1EE
// CHECK: %[[ARG:.*]] = alloca i8
// CHECK: %[[ARG_LOAD:.*]] = load i8, ptr %[[ARG]]
// CHECK: %[[CAST:.*]] = icmp ne i8 %[[ARG_LOAD]], 0
// CHECK: switch i1 %[[CAST]], label %{{.*}} [
// CHECK:   i1 false, label %
// CHECK:   i1 true, label %
// CHECK: ]
}
}
