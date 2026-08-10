// REQUIRES: x86-registered-target

// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-unknown-linux -o - -mno-tls-direct-seg-refs | FileCheck %s -check-prefix=NO-TLSDIRECT
// RUN: %clang_cc1 %s -emit-llvm -triple x86_64-unknown-linux -o - | FileCheck %s -check-prefix=TLSDIRECT

// NO-TLSDIRECT: attributes #{{[0-9]+}} = {{{.*}} "indirect-tls-seg-refs"
// TLSDIRECT-NOT: attributes #{{[0-9]+}} = {{{.*}} "indirect-tls-seg-refs"

void test1(void) {
}
