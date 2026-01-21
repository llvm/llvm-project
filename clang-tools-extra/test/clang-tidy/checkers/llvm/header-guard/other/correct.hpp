#ifndef LLVM_HEADER_GUARD_OTHER_CORRECT_HPP
#define LLVM_HEADER_GUARD_OTHER_CORRECT_HPP
#endif

// RUN: %check_clang_tidy %s llvm-header-guard correct -export-fixes=%t.yaml > %t.msg 2>&1
// RUN: FileCheck -input-file=%t.msg -check-prefix=CHECK-MSG %s
// CHECK-MSG: warning: code/includes outside of area guarded by header guard; consider moving it [llvm-header-guard]
