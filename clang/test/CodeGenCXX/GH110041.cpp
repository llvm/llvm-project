// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s

// Regression test for https://github.com/llvm/llvm-project/issues/110041
// (root cause shared with https://github.com/llvm/llvm-project/issues/175934:
// `&static_data_member` was value-dependent but not instantiation-dependent, so
// decltype(&b) kept its uninstantiated, dependent-flagged type in a<int>).
// This used to crash CodeGen in ConstantEmitter::tryEmitPrivate.

template <typename> struct a {
  static char const b{};
  static decltype(&b) constexpr c{&b};
};

// CHECK: @x = {{.*}}global ptr @_ZN1aIiE1bE
auto x = a<int>::c;

// Local variant: the variable's type comes from decltype(&static_local).
template <class T> int get() {
  static const int e = 42;
  decltype(&e) p = &e;
  return *p;
}

// CHECK-LABEL: define {{.*}}i32 @_Z3getIiEiv()
// CHECK: ret i32
int use() { return get<int>(); }
