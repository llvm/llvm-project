// RUN: %clang_cc1 -triple %itanium_abi_triple -emit-llvm -std=c++11 -fcxx-exceptions -fexceptions -S -emit-llvm -o - %s | FileCheck %s

namespace std {
  struct string {
    const char *p;
    string(const char *s);
    ~string();
  };
}

struct Bar {
  int a;
};

struct Foo {
  std::string c;
  Bar d[32];
};

static Foo table[] = {
  { "blerg" },
};

// CHECK: define internal void @__cxx_global_var_init
// CHECK: invoke {{.*}} @_ZNSt6stringC1EPKc(
// CHECK-NOT: unreachable
// CHECK: br label
