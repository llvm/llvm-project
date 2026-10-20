// REQUIRES: systemz-registered-target

// RUN: %clang -S -emit-llvm --target=s390x-none-zos %s -o - | FileCheck %s

#pragma csect(code, "codecsect")
#pragma csect(static, "staticcsect")

int main() { return 0; }

// CHECK: !{i32 1, !"zos_csect_code", !"codecsect"}
// CHECK: !{i32 1, !"zos_csect_static", !"staticcsect"}
