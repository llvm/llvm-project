// RUN: c-index-test -test-load-source-memory-usage none %s 2>&1 | FileCheck %s

// We would create millions of Exprs to fill out the initializer.

double data[1000000] = {0};
double data_empty_init[1000000] = {};

struct S {
 S(int);
 S();
};

S data2[1000000] = {0};
S data_empty_init2[1000000] = {};

// CHECK: TOTAL = {{.*}} ({{0|1}}.{{.*}} MBytes)
