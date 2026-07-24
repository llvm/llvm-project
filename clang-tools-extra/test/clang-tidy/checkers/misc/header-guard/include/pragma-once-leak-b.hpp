// No header guard, no pragma once.

// RUN: %check_clang_tidy %s misc-header-guard pragma-once-leak-b -export-fixes=%t.yaml \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.AllowPragmaOnce: true, \
// RUN:   }}' > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MSG %s
// CHECK-MSG: :1:1: warning: header is missing header guard [misc-header-guard]
