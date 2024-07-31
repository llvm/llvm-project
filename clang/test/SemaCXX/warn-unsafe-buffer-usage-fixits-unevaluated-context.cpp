// RUN: %clang_cc1 -std=c++20 -Wunsafe-buffer-usage \
// RUN:            -fsafe-buffer-usage-suggestions \
// RUN:            -fdiagnostics-parseable-fixits \
// RUN:            -fsyntax-only %s 2>&1 | FileCheck %s

namespace std {
  class type_info;
  class bad_cast;
  class bad_typeid;
}
using size_t = __typeof(sizeof(int));
void *malloc(size_t);

void foo(...);
int bar(int *ptr);

void uneval_context_fix_pointer_dereference() {
  int* p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  typeid(foo(*p));
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:14-[[@LINE-1]]:15}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:16-[[@LINE-2]]:16}:"[0]"
  _Generic(*p, int: 2, float: 3);
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:12-[[@LINE-1]]:13}:""
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:14-[[@LINE-2]]:14}:"[0]"
}

void uneval_context_fix_pointer_array_access() {
  int* p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  typeid(foo(p[5]));
  _Generic(p[2], int: 2, float: 3);
}

void uneval_context_fix_pointer_reference() {
  int* p = new int[10];
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:3-[[@LINE-1]]:7}:"std::span<int>"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:12-[[@LINE-2]]:12}:"{"
  // CHECK: fix-it:"{{.*}}":{[[@LINE-3]]:23-[[@LINE-3]]:23}:", 10}"

  int tmp = p[5];
  typeid(bar(p));
  // CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:15-[[@LINE-1]]:15}:".data()"
}

// The FixableGagdtes are not working in the following scenarios:
// 1. sizeof(DRE)
// 2. typeid(DRE)
// 3. __typeof(DRE)
// 4. _Generic(expr, type_1: DRE, type_2:)
// 5. decltype(DRE) var = y;
// 6. noexcept(DRE);
// This is becauste the UPC and ULC context matchers do not handle these contexts
// and almost all FixableGagdets currently depend on these matchers.

// FIXME: Emit fixits for each of the below use.
void uneval_context_fix_pointer_dereference_not_handled() {
  int* p = new int[10];
  int tmp = p[5];

  foo(sizeof(*p), sizeof(decltype(*p)));
  __typeof(*p) x;
  int *q = (int *)malloc(sizeof(*p));
  int y = sizeof(*p);
  __is_pod(__typeof(*p));
  __is_trivially_constructible(__typeof(*p), decltype(*p));
  _Generic(*p, int: 2, float: 3);
  _Generic(1, int: *p, float: 3);
  _Generic(1, int: 2, float: *p);
  decltype(*p) var = y;
  noexcept(*p);
  typeid(*p);
}

