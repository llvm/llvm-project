// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -O1 -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,EXACT
// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -O0 -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,INEXACT
// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -O1 -fvisibility=hidden -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,INEXACT
// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -O1 -fapple-kext -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,INEXACT
// RUN: %clang_cc1 -I%S %s -triple x86_64-apple-darwin10 -O1 -fno-assume-unique-vtables -emit-llvm -std=c++11 -o - | FileCheck %s --check-prefixes=CHECK,INEXACT

struct A { virtual ~A(); };
struct B final : A { };

// CHECK-LABEL: @_Z5exactP1A
B *exact(A *a) {
  // INEXACT: call {{.*}} @__dynamic_cast
  // EXACT-NOT: call {{.*}} @__dynamic_cast
  return dynamic_cast<B*>(a);
}
