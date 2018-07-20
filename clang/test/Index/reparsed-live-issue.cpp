// RUN: env CINDEXTEST_EDITING=1 LIBCLANG_DISABLE_CRASH_RECOVERY=1 c-index-test -test-load-source-reparse 2 none -remap-file-0=%S/Inputs/reparse-issue.h,%S/Inputs/reparse-issue.h-0 -remap-file-1=%S/Inputs/reparse-issue.h,%S/Inputs/reparse-issue.h-1 -- %s 2>&1 | FileCheck %s
#include "Inputs/reparse-issue.h"

// CHECK: reparse-issue.h:4:1:{1:1-1:1}: error: C++ requires a type specifier for all declarations
