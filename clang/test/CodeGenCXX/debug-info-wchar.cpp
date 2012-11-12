// RUN: %clang_cc1 -emit-llvm -g %s -o -| FileCheck %s
void foo() {
// CHECK: metadata !"wchar_t",
  const wchar_t w = L'x';
}
