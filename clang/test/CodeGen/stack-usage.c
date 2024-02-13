// REQUIRES: aarch64-registered-target

// RUN: rm -rf %t && split-file %s %t && cd %t
// RUN: %clang_cc1 -triple aarch64-unknown -I . -stack-usage-file a.su -emit-obj a.c -o a.o
// RUN: FileCheck %s < a.su

// CHECK: {{.*}}x.inc:1:bar	[[#]]	dynamic
// CHECK: a.c:2:foo	[[#]]	static
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
