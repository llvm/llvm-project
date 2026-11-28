// REQUIRES: systemz-registered-target

// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-pch -o %t %s
// RUN: %clang_cc1 -triple s390x-ibm-zos -include-pch %t -emit-llvm %s -o - | FileCheck %s

#ifndef HEADER
#define HEADER
#pragma csect(code, "codecsect")
#pragma csect(static, "staticcsect")

#else

int main() { return 0; }

// CHECK: define {{.*}} @main()
// CHECK: !{i32 1, !"zos_csect_code", !"codecsect"}
// CHECK: !{i32 1, !"zos_csect_static", !"staticcsect"}

#endif
