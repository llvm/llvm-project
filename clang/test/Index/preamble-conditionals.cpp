// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source local %s 2>&1 \
// RUN: | FileCheck %s --implicit-check-not "error:"
#ifndef FOO_H
#define FOO_H

void foo();

#endif
