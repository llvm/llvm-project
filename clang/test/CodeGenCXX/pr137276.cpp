// RUN: %clang_cc1 -std=c++20 -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

using ulong = unsigned long;
template <class... Ts>
void g(Ts... args) {
  ulong arr[3] = {ulong(args)...};
  (void)arr;
}
extern void f() {
  g(nullptr, 17);
}

// CHECK: {{^}}  store i64 0, ptr %arr, align 8{{$}}
