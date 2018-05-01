// RUN: %clang_cc1 -std=c++17 -emit-llvm -fchar8_t -triple x86_64-linux %s -o - | FileCheck %s

// CHECK: define void @_Z1fDu(
void f(char8_t c) {}

// CHECK: define weak_odr void @_Z1gIiEvDTplplcvT__ELA4_KDuELDu114EE
template<typename T> void g(decltype(T() + u8"foo" + u8'r')) {}
template void g<int>(const char8_t*);
