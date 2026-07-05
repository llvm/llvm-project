// RUN: mkdir -p %t
// RUN: touch %t/header.h
// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 2 local %s -I %t 2>&1 > %t.out.txt | FileCheck -check-prefix=STDERR %s
// RUN: FileCheck -input-file=%t.out.txt %s
// CHECK: preamble-reparse-warn-end-of-file.c:[[@LINE+6]]:6: FunctionDecl=test:[[@LINE+6]]:6
// STDERR: preamble-reparse-warn-end-of-file.c:[[@LINE+5]]:14: error: expected '}'
// STDERR: preamble-reparse-warn-end-of-file.c:[[@LINE+4]]:14: error: expected '}'

#include "header.h"

void test() {
