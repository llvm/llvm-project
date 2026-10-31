// RUN: %clang_cc1 %s -triple %itanium_abi_triple -O1 -disable-llvm-passes -emit-llvm -o - | FileCheck %s

// Test the attributes for the lambda function contains 'optnone' as result of
// the _Pragma("clang optimize off").

_Pragma("clang optimize off")

void foo(int p) {
  auto lambda = [&p]() { ++p; };
  lambda();
  // CHECK: define {{.*}} @"_ZZ3fooiENK3$_0clEv"({{.*}}) #[[LAMBDA_ATR:[0-9]+]]
}

_Pragma("clang optimize on")

// An always_inline lambda should not have noinline and optnone and should
// compile under _Pragma("clang optimize off")
_Pragma("clang optimize off")

__attribute__((always_inline)) void bar() {}
// CHECK: define {{.*}}void @_Z3barv() #[[ALWAYSINLINE:[0-9]+]]

auto lambda = []() __attribute__((always_inline)) { return 42; };
// CHECK: define {{.*}} @"_ZNK3$_1clEv"({{.*}}) #[[ALWAYSINLINE]]

int caller() {
  bar();
  return lambda();
}

// CHECK: attributes #[[LAMBDA_ATR]] = { {{.*}} optnone {{.*}} }
// CHECK: attributes #[[ALWAYSINLINE]] = {{{.*}}alwaysinline{{.*}}}
