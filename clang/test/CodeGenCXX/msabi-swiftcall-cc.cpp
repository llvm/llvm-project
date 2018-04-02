// RUN: %clang_cc1 -triple i686-unknown-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-windows-msvc -fdeclspec -emit-llvm %s -o - | FileCheck %s -check-prefix CHECK-64

void __attribute__((__swiftcall__)) f() {}
// CHECK-DAG: @"\01?f@@YSXXZ"
// CHECK-64-DAG: @"\01?f@@YSXXZ"

void (__attribute__((__swiftcall__)) *p)();
// CHECK-DAG: @"\01?p@@3P6SXXZA"
// CHECK-64-DAG: @"\01?p@@3P6SXXZEA

namespace {
void __attribute__((__swiftcall__)) __attribute__((__used__)) f() { }
}
// CHECK-DAG: @"\01?f@?A@@YSXXZ"
// CHECK-64-DAG: @"\01?f@?A@@YSXXZ"

namespace n {
void __attribute__((__swiftcall__)) f() {}
}
// CHECK-DAG: @"\01?f@n@@YSXXZ"
// CHECK-64-DAG: @"\01?f@n@@YSXXZ"

struct __declspec(dllexport) S {
  S(const S &) = delete;
  S & operator=(const S &) = delete;
  void __attribute__((__swiftcall__)) m() { }
};
// CHECK-DAG: @"\01?m@S@@QASXXZ"
// CHECK-64-DAG: @"\01?m@S@@QEASXXZ"

void f(void (__attribute__((__swiftcall__))())) {}
// CHECK-DAG: @"\01?f@@YAXP6SXXZ@Z"
// CHECK-64-DAG: @"\01?f@@YAXP6SXXZ@Z"

void __attribute__((__preserve_most__)) g() {}
// CHECK-DAG: @"\01?g@@YUXXZ"
// CHECK-64-DAG: @"\01?g@@YUXXZ"

void (__attribute__((__preserve_most__)) *q)();
// CHECK-DAG: @"\01?q@@3P6UXXZA"
// CHECK-64-DAG: @"\01?q@@3P6UXXZEA"

namespace {
void __attribute__((__preserve_most__)) __attribute__((__used__)) g() {}
}
// CHECK-DAG: @"\01?g@?A@@YUXXZ"
// CHECK-64-DAG: @"\01?g@?A@@YUXXZ"

namespace n {
void __attribute__((__preserve_most__)) g() {}
}
// CHECK-DAG: @"\01?g@n@@YUXXZ"
// CHECK-64-DAG: @"\01?g@n@@YUXXZ"

struct __declspec(dllexport) T {
  T(const T &) = delete;
  T & operator=(const T &) = delete;
  void __attribute__((__preserve_most__)) m() {}
};
// CHECK-DAG: @"\01?m@T@@QAUXXZ"
// CHECK-64-DAG: @"\01?m@T@@QEAUXXZ"

void g(void (__attribute__((__preserve_most__))())) {}
// CHECK-DAG: @"\01?g@@YAXP6UXXZ@Z"
// CHECK-64-DAG: @"\01?g@@YAXP6UXXZ@Z"

