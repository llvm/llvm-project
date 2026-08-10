// RUN: env CINDEXTEST_EDITING=1 CINDEXTEST_COMPLETION_BRIEF_COMMENTS=1 LIBCLANG_DISABLE_CRASH_RECOVERY=1 \
// RUN:   c-index-test -test-load-source-reparse 2 all -remap-file-0="%S/test.h,%S/test.h-0" -remap-file-1="%S/test.h,%S/test.h-1" -- %s

#include "test.h"

void foo() {
  bar();
}
