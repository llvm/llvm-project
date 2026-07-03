// REQUIRES: systemz-registered-target

// test default
// RUN: %clang_cc1 -triple s390x-ibm-zos -emit-llvm %s -o -\
// RUN:   | FileCheck %s -check-prefix=DEFAULT

// test the positive and negative options
// RUN: %clang_cc1 -triple s390x-ibm-zos -mzos-ppa1-name -emit-llvm %s -o -\
// RUN:   | FileCheck %s -check-prefix=EMIT-NAME
// RUN: %clang_cc1 -triple s390x-ibm-zos -mno-zos-ppa1-name -emit-llvm %s -o -\
// RUN:   | FileCheck %s -check-prefix=NOT-EMIT-NAME

// DEFAULT-NOT: attributes #0 = {{{.*}}"zos-ppa1-name"{{.*}}}
// EMIT-NAME: attributes #0 = {{{.*}}"zos-ppa1-name"="all"{{.*}}}
// NOT-EMIT-NAME: attributes #0 = {{{.*}}"zos-ppa1-name"="none"{{.*}}}

int main() {
  return 0;
}
