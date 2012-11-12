// RUN: %clang_cc1 -emit-llvm -g -triple x86_64-apple-darwin %s -o - | FileCheck %s

class base { };

template <class T> class foo : public base  {
  void operator=(const foo r) { }
};

class bar : public foo<void> { };
bar filters;

// For now check that it simply doesn't crash.
// CHECK: {{.*}}
