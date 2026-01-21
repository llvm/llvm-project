

// RUN: %check_clang_tidy %s llvm-header-guard missing -export-fixes=%t.yaml > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MSG %s
// CHECK-MSG: :1:1: warning: header is missing header guard [llvm-header-guard]
