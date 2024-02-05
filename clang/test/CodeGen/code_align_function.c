// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -x c %s %s -o - | FileCheck -check-prefix=CHECK-C %s
// RUN: %clang_cc1 -fsyntax-only -emit-llvm -x c++ -std=c++11 %s -o - | FileCheck %s --check-prefixes CHECK-CPP

// CHECK-C: define dso_local i32 @code_align_function(i32 noundef %a) #[[A0:[0-9]+]]
// CHECK-C: define dso_local i32 @code_align_function_with_aligned_loop(i32 noundef %a) #[[A1:[0-9]+]]

// CHECK-C: attributes #[[A0]] = {{.*}} "align-basic-blocks"="32"
[[clang::code_align(32)]] int code_align_function(int a) {
  if (a) {
    return 2;
  }
  return 3;
}

// CHECK-C: attributes #[[A1]] = {{.*}} "align-basic-blocks"="64"
[[clang::code_align(64)]] int code_align_function_with_aligned_loop(int a) {
  if (a) {
    return 2;
  }
  int c = 0;
  // CHECK-C: !{!"llvm.loop.align", i32 128}
  [[clang::code_align(128)]] for (int i = 0; i < a; ++i) {
    c += i;
  }
  return c;
}

#if __cplusplus >= 201103L
struct S {
  // CHECK-CPP: @_ZN1S19code_align_functionILi1EEEii({{.*}}) #[[A2:[0-9]+]]
  // CHECK-CPP: attributes #[[A2]] = {{.*}} "align-basic-blocks"="16"
  template <int A>
  [[clang::code_align(16)]] int code_align_function(int a) {
    if (a) {
      return 2;
    }
    return 3;
  }
};

template int S::code_align_function<1>(int);
#endif
