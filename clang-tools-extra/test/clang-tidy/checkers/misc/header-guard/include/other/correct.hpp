#ifndef OTHER_CORRECT_HPP
#define OTHER_CORRECT_HPP
// RUN: %check_clang_tidy %s misc-header-guard correct -export-fixes=%t.yaml > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MSG %s
// CHECK-MSG-NOT: warning:
#endif
