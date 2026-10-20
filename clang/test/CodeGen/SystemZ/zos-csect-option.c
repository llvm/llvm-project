// REQUIRES: systemz-registered-target

// RUN: %clang -S -emit-llvm --target=s390x-none-zos %s -o outputfile.ll && cat outputfile.ll | FileCheck %s --check-prefix=CHECK-NOCSECT
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect %s -o outputfile.ll && cat outputfile.ll | FileCheck %s
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect= %s -o outputfile.ll && cat outputfile.ll | FileCheck %s
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect=abc %s -o outputfile.ll && cat outputfile.ll | FileCheck %s --check-prefix=CHECK-QUALIFIER
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mno-csect %s -o outputfile.ll && cat outputfile.ll | FileCheck %s --check-prefix=CHECK-NOCSECT
// RUN: %clang -S -emit-llvm --target=s390x-linux-gnu %s -o outputfile.ll && cat outputfile.ll | FileCheck %s --check-prefix=CHECK-NOCSECT

int main() { return 0; }

// CHECK: !{i32 1, !"zos_csect_code", !"outputfile#C"}
// CHECK: !{i32 1, !"zos_csect_static", !"outputfile#S"}

// CHECK-QUALIFIER: !{i32 1, !"zos_csect_code", !"abc#outputfile#C"}
// CHECK-QUALIFIER: !{i32 1, !"zos_csect_static", !"abc#outputfile#S"}

// CHECK-NOCSECT-NOT: zos_csect_code
// CHECK-NOCSECT-NOT: zos_csect_static
