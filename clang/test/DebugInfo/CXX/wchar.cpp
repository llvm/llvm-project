// RUN: %clang_cc1 -emit-llvm -debug-info-kind=limited %s -o -| FileCheck %s
void foo() {
// CHECK: !DIBasicType(name: "wchar_t"
  const wchar_t w = L'x';
}
