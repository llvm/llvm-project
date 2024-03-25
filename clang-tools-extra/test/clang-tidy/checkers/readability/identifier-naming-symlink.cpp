// Specify `-std=c++20` to run test only once becuase test expects changes
// in the header file so it fails if runs multiple times with different
// `-std` flags as check_clang_tidy doesn by default.
//
// RUN: rm -rf %T/symlink
// RUN: cp -r %S/Inputs/identifier-naming/symlink %T/symlink
// RUN: mkdir -p %T/symlink/build
// RUN: ln -s %T/symlink/include/test.h %T/symlink/build/test.h
// RUN: %check_clang_tidy -std=c++20 %s readability-identifier-naming %t -- --header-filter="test.h" --config-file=%S/Inputs/identifier-naming/symlink/include/.clang-tidy -- -I %T/symlink/build
// UNSUPPORTED: system-windows

#include "test.h"

int foo() {
    return global_const;
    // CHECK-MESSAGES: warning: invalid case style for global constant 'global_const' [readability-identifier-naming]
    // CHECK-FIXES: return kGlobalConst;
}
