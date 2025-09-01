// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir -p %t/dir1/dir2
// RUN: echo 'class A { A(int); };' > %t/dir1/header.h
// RUN: ln -s %t/dir1/header.h %t/dir1/header_alias.h
//
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='dir1/dir2/\.\./header_alias\.h' %s -- -I %t 2>&1 | FileCheck --check-prefix=CHECK_HEADER_ALIAS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='dir1/dir2/\.\./header_alias\.h' -quiet %s -- -I %t 2>&1 | FileCheck --check-prefix=CHECK_HEADER_ALIAS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='header_alias\.h' %s -- -I %t 2>&1 | FileCheck --check-prefix=CHECK_HEADER_ALIAS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='header_alias\.h' -quiet %s -- -I %t 2>&1 | FileCheck --check-prefix=CHECK_HEADER_ALIAS %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='header\.h' %s -- -I %t 2>&1 | FileCheck --check-prefix=CHECK_HEADER %s
// RUN: clang-tidy -checks='-*,google-explicit-constructor' -header-filter='header\.h' -quiet %s -- -I %t 2>&1 | FileCheck --check-prefix=CHECK_HEADER --allow-empty %s

// Check that `-header-filter` operates on the same file paths as paths in
// diagnostics printed by ClangTidy.
#include "dir1/dir2/../header_alias.h"
// CHECK_HEADER_ALIAS: dir1/dir2/../header_alias.h:1:11: warning: single-argument constructors
// CHECK_HEADER-NOT: warning:
