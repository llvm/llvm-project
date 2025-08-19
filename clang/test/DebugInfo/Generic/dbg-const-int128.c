// RUN: %clang_cc1 -triple x86_64-unknown-linux -emit-llvm -debug-info-kind=limited  %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix-xcoff -emit-llvm -debug-info-kind=limited  %s -o - | FileCheck %s
// CHECK: !DIGlobalVariable({{.*}}
// CHECK-NOT: expr:

static const __uint128_t ro = 18446744073709551615;

void bar(__uint128_t);
void foo(void) { bar(ro); }
