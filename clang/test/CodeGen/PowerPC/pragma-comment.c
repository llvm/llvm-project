// RUN: split-file %s %t

// RUN: %clang_cc1 %t/pragma-copyright.c -triple powerpc-ibm-aix   -emit-llvm -disable-llvm-passes -o - | FileCheck %s  --check-prefix=CHECK-COPYRIGHT
// RUN: %clang_cc1 %t/pragma-copyright.c -triple powerpc64-ibm-aix -emit-llvm -disable-llvm-passes -o - | FileCheck %s  --check-prefix=CHECK-COPYRIGHT

// RUN: %clang_cc1 %t/operator-pragma-copyright.c -triple powerpc-ibm-aix   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-OPERATOR
// RUN: %clang_cc1 %t/operator-pragma-copyright.c -triple powerpc64-ibm-aix -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-OPERATOR

//--- pragma-copyright.c
#pragma comment(copyright, "@(#) Hello, " " world\n\t\"quoted\"")

int main() { return 0; }

// CHECK-COPYRIGHT: !comment_string.loadtime = !{![[COPYRIGHT:[0-9]+]]}
// CHECK-COPYRIGHT: ![[COPYRIGHT]] = !{!"@(#) Hello, world\0A\09\22quoted\22"}

//--- operator-pragma-copyright.c
_Pragma("comment(copyright, \"IBM Copyright Pragma Operator\")")
void foo() {}

// CHECK-OPERATOR: !comment_string.loadtime = !{![[COPYRIGHT:[0-9]+]]}
// CHECK-OPERATOR: ![[COPYRIGHT]] = !{!"IBM Copyright Pragma Operator"}
