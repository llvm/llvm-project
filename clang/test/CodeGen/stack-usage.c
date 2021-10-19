// REQUIRES: aarch64-registered-target

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: %clang_cc1 -triple aarch64-unknown -stack-usage-file b.su -emit-obj %s -o b.o
// RUN: FileCheck %s < b.su

// CHECK: stack-usage.c:[[#@LINE+1]]:foo	{{[0-9]+}}	static
int foo() {
  char a[8];

  return 0;
}

// CHECK: stack-usage.c:[[#@LINE+1]]:bar	{{[0-9]+}}	dynamic
int bar(int len) {
  char a[len];

  return 1;
}
