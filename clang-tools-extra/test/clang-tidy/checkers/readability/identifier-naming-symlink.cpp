// Specify `-std=c++20` to run test only once becuase test expects changes
// in the header file so it fails if runs multiple times with different
// `-std` flags as check_clang_tidy doesn by default.
//
// RUN: rm -rf %t.dir
// RUN: mkdir -p %t.dir
// RUN: cp -r %S/Inputs/identifier-naming/symlink %t.dir/symlink
// RUN: mkdir -p %t.dir/symlink/build
// RUN: ln -s %t.dir/symlink/include/test.h %t.dir/symlink/build/test.h
// RUN: %check_clang_tidy -std=c++20 %s readability-identifier-naming %t.dir -- --header-filter="test.h" --config-file=%S/Inputs/identifier-naming/symlink/include/.clang-tidy -- -I %t.dir/symlink/build
// UNSUPPORTED: system-windows

#include "test.h"

int foo() {
    return global_const;
    // CHECK-MESSAGES: warning: invalid case style for global constant 'global_const' [readability-identifier-naming]
    // CHECK-FIXES: return kGlobalConst;
}
