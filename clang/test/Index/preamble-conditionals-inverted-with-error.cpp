// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source-reparse 5 \
// RUN:                                       local -std=c++14 %s 2>&1 \
// RUN: | FileCheck %s
#ifdef FOO_H

void foo();

// CHECK: preamble-conditionals-inverted-with-error.cpp:4:2: error: unterminated conditional directive
