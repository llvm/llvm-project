// REQUIRES: systemz-registered-target

// test default
// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-llvm %s -o -\
// RUN:   | FileCheck %s -check-prefix=EMIT

// test the positive and negative options
// RUN: %clang_cc1 -triple s390x-ibm-zos -mzos-ppa1-name -emit-llvm %s -o -\
// RUN:   | FileCheck %s -check-prefix=EMIT
// RUN: %clang_cc1 -triple s390x-ibm-zos -mno-zos-ppa1-name -emit-llvm %s -o -\
// RUN:   | FileCheck %s -check-prefix=NOEMIT

// EMIT-NOT: attributes #0 = {{{.*}}"zos-ppa1-name"{{.*}}}
// NOEMIT: attributes #0 = {{{.*}}"zos-ppa1-name"{{.*}}}

int main() {
  return 0;
}
