// RUN: %clang_cc1 -triple m68k-linux-gnu -emit-llvm -o - %s | FileCheck %s

class A {
public:
// CHECK: define{{.*}} m68k_rtdcc void @_ZN1A6memberEv
  void __attribute__((m68k_rtd)) member() {}
};

void test() {
  A a;
  a.member();

// CHECK: define{{.*}} m68k_rtdcc void @"_ZZ4testvENK3$_0clEi"
  auto f = [](int b) __attribute__((m68k_rtd)) {};
  f(87);
};
