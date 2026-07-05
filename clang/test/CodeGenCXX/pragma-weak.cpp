// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm %s -o - | FileCheck %s

#pragma weak zex
int zex;
// GCC produces a weak symbol for this because it matches mangled names.
// Different c++ ABIs may or may not mangle this, so we produce a strong
// symbol.
// CHECK: @zex = {{(dso_local )?}}global i32

#pragma weak foo
struct S {  void foo(); };
void S::foo() {}
// CHECK-LABEL: define {{.*}}void @_ZN1S3fooEv(

#pragma weak zed
namespace bar {  void zed() {} }
// CHECK-LABEL: define {{.*}}void @_ZN3bar3zedEv(

#pragma weak bah
void bah() {}
// CHECK-LABEL: define {{.*}}void @_Z3bahv(

#pragma weak baz
extern "C" void baz() {}
// CHECK-LABEL: define weak {{.*}}void @baz(

#pragma weak _Z3baxv
void bax() {}
// GCC produces a weak symbol for this one, but it doesn't look like a good
// idea to expose the mangling to the pragma unless we really have to.
// CHECK-LABEL: define {{.*}}void @_Z3baxv(
