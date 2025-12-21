// REQUIRES: aarch64-registered-target
// REQUIRES: x86-registered-target

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: %clang_cc1 -triple aarch64-unknown -O0 -I . -stack-usage-file a-O0.su -emit-obj a.c -o a-O0.o
// RUN: FileCheck %s < a-O0.su -check-prefix=O0
// RUN: %clang_cc1 -triple aarch64-unknown -O3 -I . -stack-usage-file a-O3.su -emit-obj a.c -o a-O3.o
// RUN: FileCheck %s < a-O3.su -check-prefix=O3
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -O3 -I . -stack-usage-file a-x86.su -emit-obj a.c -o a-x86.o
// RUN: FileCheck %s < a-x86.su -check-prefix=x86

// O0: {{.*}}x.inc:1:bar	[[#]]	dynamic
// O0: a.c:2:foo	[[#]]	static

// O3: {{.*}}x.inc:1:bar	[[#]]	static
// O3: a.c:2:foo	[[#]]	static

// For x86 stack usage must be greater than zero
// x86: {{.*}}x.inc:1:bar {{[1-9][0-9]*}} static
// x86: a.c:2:foo {{[1-9][0-9]*}} static

//--- a.c
#include "x.inc"
int foo() {
  char a[8];

  return 0;
}

//--- x.inc
int bar(int len) {
  char a[len];

  return 1;
}
