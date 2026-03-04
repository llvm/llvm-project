#ifndef HERE_IS_SOMETHING_WRONG_HPP
#define HERE_IS_SOMETHING_WRONG_HPP
#endif

// RUN: %check_clang_tidy %s misc-header-guard wrong -export-fixes=%t.yaml > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MSG %s
// CHECK-MSG: warning: header guard does not follow preferred style [misc-header-guard]
