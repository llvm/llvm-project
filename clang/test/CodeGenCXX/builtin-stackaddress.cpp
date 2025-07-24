// RUN: %clang -target x86_64 -S -emit-llvm %s -o - | llvm-cxxfilt | FileCheck %s --check-prefix=llvm
// RUN: %clang -target x86_64 -S %s -o - | llvm-cxxfilt | FileCheck %s --check-prefix=x64

extern void f(int, int, int, long, long, long, long, long, long, long, long);

struct S {
  void *a();
};

// llvm-LABEL: define {{[^@]+}} @S::a()
// llvm:       call   {{[^@]+}} @llvm.stackaddress.p0()
//
// x64-LABEL: S::a():
// x64:       movq  %rsp, %rax
void *S::a() {
  void *p = __builtin_stack_address();
  f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
  return p;
}

// llvm-LABEL: define {{[^@]+}} @two()
// llvm:       call   {{[^@]+}} @"two()::$_0::operator()() const"
//
// llvm-LABEL: define {{[^@]+}} @"two()::$_0::operator()() const"
// llvm:       call   {{[^@]+}} @llvm.stackaddress.p0()
//
// x64-LABEL: two()::$_0::operator()() const:
// x64:       movq  %rsp, %rax
void *two() {
  auto l = []() {
    void *p = __builtin_stack_address();
    f(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
    return p;
  };
  return l();
}
