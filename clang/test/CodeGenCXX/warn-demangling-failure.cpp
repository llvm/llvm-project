// RUN: %clang_cc1 -emit-llvm -fdemangling-failures -triple %itanium_abi_triple -o - %s | FileCheck %s
template <class F> void parallel_loop(F &&f) { f(0); }

//CHECK-LABEL: @main
int main() {
  int x;
  parallel_loop([&](auto y) { // expected-warning {{cannot demangle the name '_ZZ4mainENK3$_0clIiEEDaT_'}}
#pragma clang __debug captured
    {
      x = y;
    };
  });
}
