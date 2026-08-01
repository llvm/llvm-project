// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -emit-llvm -o - %s | FileCheck %s

class Foo;
typedef __SIZE_TYPE__ size_t;

namespace std {
enum class align_val_t : size_t {};
template <class T> struct type_identity {
  typedef T type;
};
} // namespace std

template <class T>
void operator delete(std::type_identity<T>, void *, size_t, std::align_val_t);

// CHECK-LABEL: define{{.*}} void @_Z1fP3Foo(
// CHECK: call void @_ZdlPv(
// CHECK-NOT: call void @_ZdlI3FooEvSt13type_identityIT_EPvmSt11align_val_t(
void f(Foo *o) {
  delete o;
}
