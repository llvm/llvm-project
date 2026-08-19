#pragma once

// RUN: %check_clang_tidy %s misc-header-guard pragma-once -export-fixes=%t.1.yaml > %t.1.msg 2>&1
// RUN: FileCheck -input-file=%t.1.msg -check-prefix=CHECK-MSG1 %s
// CHECK-MSG1: pragma-once.hpp:1:1: warning: use include guards instead of 'pragma once' [misc-header-guard]

// RUN: %check_clang_tidy %s misc-header-guard pragma-once \
// RUN:   --config='{CheckOptions: { \
// RUN:     misc-header-guard.AllowPragmaOnce: true, \
// RUN:   }}' > %t.2.msg 2>&1
