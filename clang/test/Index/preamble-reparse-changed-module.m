// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir -p %t/mod
// RUN: touch %t/empty.h
// RUN: cp %S/Inputs/preamble-reparse-changed-module/module.modulemap %t/mod
// RUN: cp %S/Inputs/preamble-reparse-changed-module/head.h %t/mod

// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_EXECUTE_COMMAND="cp %S/Inputs/preamble-reparse-changed-module/new-head.h %t/mod/head.h" CINDEXTEST_EXECUTE_AFTER_TRIAL=1 \
// RUN:     c-index-test -test-load-source-reparse 3 local %s -I %t -I %t/mod -fmodules -fmodules-cache-path=%t/mcp 2>&1 | FileCheck %s

// CHECK-NOT: warning:

#include "empty.h"
@import mod;

void test(I *o) {
  [o call_me_new];
}
