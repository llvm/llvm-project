// RUN: %clang_cc1 -triple powerpc64-ibm-aix -emit-llvm -o - %s -msave-reg-params | FileCheck -check-prefix=SAVE %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix -emit-llvm -o - %s -msave-reg-params | FileCheck -check-prefix=SAVE %s
// RUN: %clang_cc1 -triple powerpc64-ibm-aix -emit-llvm -o - %s | FileCheck -check-prefix=NOSAVE %s
// RUN: %clang_cc1 -triple powerpc-ibm-aix -emit-llvm -o - %s | FileCheck -check-prefix=NOSAVE %s

void bar(int);
void foo(int x) { bar(x); }

// SAVE: attributes #{{[0-9]+}} = { {{.+}} "save-reg-params" {{.+}} }
// NOSAVE-NOT: "save-reg-params"
