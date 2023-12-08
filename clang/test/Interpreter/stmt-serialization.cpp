// RUN: rm -rf %t
// RUN: %clang_cc1 -std=c++20 -fincremental-extensions -fmodules-cache-path=%t \
// RUN:            -x c++ %s -verify
// expected-no-diagnostics

#pragma clang module build TopLevelStmt
module TopLevelStmt { module Statements {} }
#pragma clang module contents

#pragma clang module begin TopLevelStmt.Statements
extern "C" int printf(const char*,...);
int i = 0;
i++;
#pragma clang module end /*TopLevelStmt.Statements*/
#pragma clang module endbuild /*TopLevelStmt*/

#pragma clang module import TopLevelStmt.Statements

printf("Value of i is '%d'", i);
