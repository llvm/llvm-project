// RUN: %clang_cc1 -triple arm64-apple-ios -fptrauth-calls -fptrauth-returns -fptrauth-intrinsics -emit-llvm -std=c++11 -fexceptions -fcxx-exceptions -o - %s | FileCheck %s

void foo1();

void test_terminate() noexcept {
  foo1();
}

// CHECK: define void @_ZSt9terminatev() #[[ATTR4:.*]] {

namespace std {
  void terminate() noexcept {
  }
}

// CHECK: attributes #[[ATTR4]] = {{{.*}}"ptrauth-calls" "ptrauth-returns"{{.*}}}
