// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin %s -o - | FileCheck %s %}

class base { };

template <class T> class foo : public base  {
  void operator=(const foo r) { }
};

class bar : public foo<void> { };
bar filters;

// For now check that it simply doesn't crash.
// CHECK: {{.*}}
