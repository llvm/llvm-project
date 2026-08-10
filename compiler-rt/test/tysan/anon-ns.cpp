// RUN: %clangxx_tysan -O0 %s -c -o %t.o
// RUN: %clangxx_tysan -O0 %s -DPMAIN -c -o %tm.o
// RUN: %clangxx_tysan -O0 %t.o %tm.o -o %t
// RUN: %run %t >%t.out 2>&1
// RUN: FileCheck %s < %t.out

#include <iostream>

// This test demonstrates that the types from anonymous namespaces are
// different in different translation units (while the char* type is the same).

namespace {
struct X {
  X(int i, int j) : a(i), b(j) {}
  int a;
  int b;
};
} // namespace

#ifdef PMAIN
void foo(void *context, int i);
char fbyte(void *context);

int main() {
  X x(5, 6);
  foo((void *)&x, 8);
  std::cout << "fbyte: " << fbyte((void *)&x) << "\n";
}
#else
void foo(void *context, int i) {
  X *x = (X *)context;
  x->b = i;
  // CHECK: ERROR: TypeSanitizer: type-aliasing-violation
  // CHECK: WRITE of size 4 at {{.*}} with type int (in (anonymous namespace)::X at offset 4) accesses an existing object of type int (in (anonymous namespace)::X at offset 4)
  // CHECK: {{#0 0x.* in foo\(void\*, int\) .*anon-ns.cpp:}}[[@LINE-3]]
}

char fbyte(void *context) { return *(char *)context; }
#endif

// CHECK-NOT: ERROR: TypeSanitizer: type-aliasing-violation
