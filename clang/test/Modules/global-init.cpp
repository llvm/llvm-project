// RUN: rm -rf %t
// RUN: mkdir %t
//
// RUN: echo '#pragma once' > %t/a.h
// RUN: echo 'struct A { A() {} int f() const; } const a;' >> %t/a.h
//
// RUN: echo '#include "a.h"' > %t/b.h
//
// RUN: echo 'module M { module b { header "b.h" export * } module a { header "a.h" export * } }' > %t/map
//
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t -fmodule-map-file=%t/map -I%t %s -emit-llvm -o - -triple %itanium_abi_triple | FileCheck %s

#include "b.h"

// CHECK: @_ZL1a = internal global
// CHECK: call {{.*}} @_ZN1AC1Ev({{.*}}@_ZL1a
// CHECK: call {{.*}} @_ZNK1A1fEv({{.*}}@_ZL1a
// CHECK: store {{.*}} @x
int x = a.f();
