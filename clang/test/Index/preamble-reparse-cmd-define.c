// RUN: c-index-test -write-pch %t.h.pch %s.h
// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_REMAP_AFTER_TRIAL=1 c-index-test -test-load-source-reparse 3 local \ 
// RUN:           "-remap-file=%s;%s.remap" %s -include %t.h -D CMD_MACRO=1 2>&1 | FileCheck %s

// CHECK-NOT: error:

int foo() {
  return x;
}
