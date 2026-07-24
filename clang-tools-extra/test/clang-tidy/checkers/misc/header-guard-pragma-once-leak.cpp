// Regression test: pragma-once state must be tracked per file, not once for
// the whole translation unit. Otherwise a `#pragma once` in one header
// (allowed via AllowPragmaOnce) would suppress the "missing header guard"
// diagnostic for an unrelated header included later in the same TU.
#include "header-guard/include/pragma-once-leak-a.hpp"
#include "header-guard/include/pragma-once-leak-b.hpp"

// RUN: %check_clang_tidy %s misc-header-guard %t -export-fixes=%t.yaml \
// RUN:   --header-filter=.* \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.AllowPragmaOnce: true, \
// RUN:   }}' -- -I%S > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MSG %s

// CHECK-MSG-NOT: pragma-once-leak-a.hpp{{.*}}warning:
// CHECK-MSG: pragma-once-leak-b.hpp:1:1: warning: header is missing header guard [misc-header-guard]
