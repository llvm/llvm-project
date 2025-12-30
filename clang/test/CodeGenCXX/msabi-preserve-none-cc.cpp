// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple aarch64-unknown-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s

void __attribute__((__preserve_none__)) f() {}
// CHECK-DAG: @"?f@@YVXXZ"

void (__attribute__((__preserve_none__)) *p)();
// CHECK-DAG: @"?p@@3P6VXXZEA

namespace {
void __attribute__((__preserve_none__)) __attribute__((__used__)) f() { }
}
// CHECK-DAG: @"?f@?A0x{{[^@]*}}@@YVXXZ"

namespace n {
void __attribute__((__preserve_none__)) f() {}
}
// CHECK-DAG: @"?f@n@@YVXXZ"

struct __declspec(dllexport) S {
  S(const S &) = delete;
  S & operator=(const S &) = delete;
  void __attribute__((__preserve_none__)) m() { }
};
// CHECK-DAG: @"?m@S@@QEAVXXZ"

void f(void (__attribute__((__preserve_none__))())) {}
// CHECK-DAG: @"?f@@YAXP6VXXZ@Z"
