// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm -o - %s | FileCheck %s

// GH95154: reopening ns must preserve v1 in its members' mangled names.
inline namespace v1 {
namespace ns {}
} // namespace v1

namespace ns {
// CHECK-LABEL: define {{.*}} @_ZN2v12ns1fEv(
void f() {}
} // namespace ns

// A qualified definition must preserve v1 in X's mangled name, as required
// by libstdc++'s filesystem::_Dir definition.
namespace qualified {
inline namespace v1 {
struct X;
} // namespace v1
} // namespace qualified
namespace alias = qualified;
struct alias::X {};

// CHECK-LABEL: define {{.*}} @_Z1fPN9qualified2v11XE(
void f(alias::X *) {}
