// REQUIRES: systemz-registered-target
// RUN: %clang_cc1 -emit-llvm -o- -triple s390x-ibm-zos %s |FileCheck %s
// RUN: %clang_cc1 -emit-llvm -o- -triple s390x-ibm-zos -mno-zos-ppa1-name %s |FileCheck --check-prefix=NONAME %s
void foo() {
}

void bar() asm("OtherName");

void bar() {
}

// CHECK:define void @foo() #[[a1:[0-9]+]]
// CHECK:define void @OtherName() #[[a2:[0-9]+]]
// CHECK-NOT:attributes #[[a1]] = {{{.*}}"zos-ppa1-name"{{.*}}}
// CHECK: attributes #[[a2]] = {{{.*}}"zos-ppa1-name"="bar"{{.*}}}


// NONAME:define void @foo() #[[a1:[0-9]+]]
// NONAME:define void @OtherName() #[[a1]]
// NONAME:attributes #[[a1]] = {{{.*}}"zos-ppa1-name"{{.*}}}



