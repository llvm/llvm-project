// RUN: %clang_cc1 -x objective-c++ -triple x86_64-apple-darwin10 -emit-llvm -fcxx-exceptions -fexceptions -fobjc-exceptions -o - %s | FileCheck %s

namespace test0 {
  void foo() {
    try {
      throw 0;
    } catch (int e) {
      return;
    }
  }
// CHECK: define{{.*}} void @_ZN5test03fooEv() #0 personality ptr @__gxx_personality_v0
}
