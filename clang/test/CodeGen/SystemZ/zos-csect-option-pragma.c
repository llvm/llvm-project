// REQUIRES: systemz-registered-target

// RUN: %clang -S -emit-llvm --target=s390x-none-zos %s -o outputfile2.ll && cat outputfile2.ll | FileCheck %s --check-prefix=CHECK-NOCSECT
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect %s -o outputfile2.ll && cat outputfile2.ll | FileCheck %s
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect= %s -o outputfile2.ll && cat outputfile2.ll | FileCheck %s
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect=abc %s -o outputfile2.ll && cat outputfile2.ll | FileCheck %s --check-prefix=CHECK-QUALIFIER
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mno-csect %s -o outputfile2.ll && cat outputfile2.ll | FileCheck %s --check-prefix=CHECK-NOCSECT
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect %s && cat zos-csect-option-pragma.ll | FileCheck %s --check-prefix=CHECK-NOOUTFILE
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect= %s && cat zos-csect-option-pragma.ll | FileCheck %s --check-prefix=CHECK-NOOUTFILE
// RUN: %clang -S -emit-llvm --target=s390x-none-zos -mcsect=cde %s && cat zos-csect-option-pragma.ll | FileCheck %s --check-prefix=CHECK-NOOUTFILE-QUALIFIER

#pragma csect(code, "csectcode")

int main() { return 0; }

// CHECK: !{i32 1, !"zos_csect_code", !"csectcode"}
// CHECK: !{i32 1, !"zos_csect_static", !"outputfile2#S"}

// CHECK-QUALIFIER: !{i32 1, !"zos_csect_code", !"csectcode"}
// CHECK-QUALIFIER: !{i32 1, !"zos_csect_static", !"abc#outputfile2#S"}

// CHECK-NOCSECT: !{i32 1, !"zos_csect_code", !"csectcode"}
// CHECK-NOCSECT-NOT: zos_csect_static

// CHECK-NOOUTFILE: !{i32 1, !"zos_csect_code", !"csectcode"}
// CHECK-NOOUTFILE: !{i32 1, !"zos_csect_static", !"zos-csect-option-pragma#S"}

// CHECK-NOOUTFILE-QUALIFIER: !{i32 1, !"zos_csect_code", !"csectcode"}
// CHECK-NOOUTFILE-QUALIFIER: !{i32 1, !"zos_csect_static", !"cde#zos-csect-option-pragma#S"}
