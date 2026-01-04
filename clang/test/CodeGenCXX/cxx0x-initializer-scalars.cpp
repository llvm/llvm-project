// RUN: %clang_cc1 -std=c++11 -triple x86_64-linux -emit-llvm -o - %s | FileCheck %s

void f()
{
  // CHECK: store i32 0
  int i{};
}


namespace GH116440 {
void f() {
  void{};
  void();
}

// CHECK: define{{.*}} void @_ZN8GH1164401fEv()
// CHECK-NEXT: entry
// CHECK-NEXT: ret void
}
