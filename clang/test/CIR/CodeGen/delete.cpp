// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -fclangir -mconstructor-aliases -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

typedef __typeof(sizeof(int)) size_t;

namespace test1 {
  struct A { void operator delete(void*,size_t); int x; };
  void a(A *x) {
    delete x;
  }
  // CHECK: cir.func @_ZN5test11aEPNS_1AE

  // CHECK: %[[CONST:.*]] = cir.const(#cir.int<4> : !u64i) : !u64i
  // CHECK: cir.call @_ZN5test11AdlEPvm({{.*}}, %[[CONST]])
}