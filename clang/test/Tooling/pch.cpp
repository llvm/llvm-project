// This is a regression test for handling of stat caches within the tooling
// infrastructure. This test reproduces the problem under valgrind:

// First, create a pch that we can later load. Loading the pch will insert
// a stat cache into the FileManager:
// RUN: %clang -x c++-header %S/Inputs/pch.h -o %t1

// Use the generated pch and enforce a subsequent stat miss by using
// the test file with an unrelated include as second translation unit.
// Test for an non-empty file after clang-check is executed.
// RUN: clang-check -ast-dump "%S/Inputs/pch.cpp" "%s" -- -include-pch %t1 -I "%S" -c >%t2 2>&1
// RUN: test -s %t2

#include "Inputs/pch-fail.h"
