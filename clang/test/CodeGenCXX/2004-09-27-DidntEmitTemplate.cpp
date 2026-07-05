// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// This is a testcase for LLVM PR445, which was a problem where the
// instantiation of callDefaultCtor was not being emitted correctly.

// CHECK-NOT: declare{{.*}}callDefaultCtor
struct Pass {};

template<typename PassName>
Pass *callDefaultCtor() { return new Pass(); }

void foo(Pass *(*C)());

struct basic_string {
  bool empty() const { return true; }
};


bool foo2(basic_string &X) {
  return X.empty();
}
void baz() { foo(callDefaultCtor<Pass>); }
