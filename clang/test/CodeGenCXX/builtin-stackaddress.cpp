// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | llvm-cxxfilt | FileCheck %s --check-prefixes=COMMON,NO-OPT
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -O3 -o - | llvm-cxxfilt | FileCheck %s --check-prefixes=COMMON,OPT

struct S {
  void *a();
};

// COMMON-LABEL: @S::a()
// COMMON: call ptr @llvm.stackaddress.p0()
void *S::a() {
  return __builtin_stack_address();
}

// COMMON-LABEL: define {{[^@]+}} @two()
void *two() {

  // The compiler is allowed to inline a function calling `__builtin_stack_address`.
  //
  // OPT-NOT: define {{[^@]+}} @"two()::$_0::operator()() const"
  // OPT: call {{[^@]+}} @llvm.stackaddress.p0()
  //
  // NO-OPT-DAG: define {{[^@]+}} @"two()::$_0::operator()() const"
  // NO-OPT-DAG: call {{[^@]+}} @"two()::$_0::operator()() const"
  auto l = []() { return __builtin_stack_address(); };
  return l();
}
