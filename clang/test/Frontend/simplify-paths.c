// REQUIRES: shell

// RUN: rm -rf %t
// RUN: mkdir -p %t/a/b/
// RUN: echo 'void x;' > %t/test.h
// RUN: echo 'const void x;' > %t/header_with_a_really_long_name.h
// RUN: ln -s %t/header_with_a_really_long_name.h %t/a/shorter_name.h
//
// RUN: %clang_cc1 -fsyntax-only -I %t %s 2> %t/output.txt || true
// RUN: cat %t/output.txt | FileCheck %s

// Check that we strip '..' by canonicalising the path...
#include "a/b/../../test.h"
// CHECK: simplify-paths.c.tmp/test.h:1:6: error: variable has incomplete type 'void'

// ... but only if the resulting path is actually shorter.
#include "a/b/../shorter_name.h"
// CHECK: simplify-paths.c.tmp/a/b/../shorter_name.h:1:12: error: variable has incomplete type 'const void'
