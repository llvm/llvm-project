// RUN: %clang_cc1 %s -emit-llvm -o - -mstackrealign | FileCheck %s -check-prefix=REALIGN
// RUN: %clang_cc1 %s -emit-llvm -o - | FileCheck %s -check-prefix=NO-REALIGN

// REALIGN: attributes #{{[0-9]+}} = {{{.*}} "stackrealign"
// NO-REALIGN-NOT: attributes #{{[0-9]+}} = {{{.*}} "stackrealign"

void test1(void) {
}
