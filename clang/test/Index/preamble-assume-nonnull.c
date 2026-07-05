// RUN: env CINDEXTEST_EDITING=1 c-index-test -test-load-source local %s 2>&1 \
// RUN: | FileCheck %s --implicit-check-not "error:"

#pragma clang assume_nonnull begin
void foo(int *x);
#pragma clang assume_nonnull end
