// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -emit-llvm -o - %s | FileCheck %s

// GH95154: reopening a namespace through an inline namespace must not change
// the namespace's semantic parent, including for newly declared members.
namespace reopen {
inline namespace v1 {
namespace ns {
struct Y;
}
}

// CHECK-DAG: define {{.*}} @_ZN6reopen6beforeEPNS_2v12ns1YE(
void before(v1::ns::Y *) {}

namespace ns {
struct X {};
struct Y {};

// CHECK-DAG: define {{.*}} @_ZN6reopen2v12ns1fEv(
void f() {}
// CHECK-DAG: @_ZN6reopen2v12ns5valueE =
int value;
}

// CHECK-DAG: define {{.*}} @_ZN6reopen5afterEPNS_2v12ns1YE(
void after(v1::ns::Y *) {}
// CHECK-DAG: define {{.*}} @_ZN6reopen4takeEPNS_2v12ns1XE(
void take(v1::ns::X *) {}
}

// A qualified class definition can find its declaration through an inline
// namespace, as in libstdc++'s filesystem::_Dir definition.
namespace qualified {
inline namespace v1 {
struct X;
}
}
namespace alias = qualified;

// CHECK-DAG: define {{.*}} @_Z6beforePN9qualified2v11XE(
void before(alias::X *) {}

struct alias::X {
  void member() {}
};

// CHECK-DAG: define {{.*}} @_Z5afterPN9qualified2v11XE(
void after(alias::X *p) { p->member(); }
// CHECK-DAG: define {{.*}} @_ZN9qualified2v11X6memberEv(

// An ordinary namespace must keep its existing mangling.
namespace ordinary {
struct X;
}
struct ordinary::X {};
// CHECK-DAG: define {{.*}} @_Z4takePN8ordinary1XE(
void take(ordinary::X *) {}
