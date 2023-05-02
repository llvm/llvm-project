// RUN: c-index-test core -scan-deps -working-dir %S -- clang_tool -Dmz -mllvm -asan-instrumentation-with-call-threshold=0 -mllvm -asan-instrumentation-with-call-threshold=0 %s -I %S/Inputs | FileCheck %s

#ifdef mz
#include "header.h"
#endif

// CHECK: file-deps:
// CHECK-NEXT: mllvm-double-option-error-c-api.c
// CHECK-NEXT: Inputs/header.h
