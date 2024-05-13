// Verifies that paths are resolved relatively to the directory specified in the
// compilation database.
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: echo "[{\"directory\":\"%t\",\"command\":\"clang -c test.cpp -I.\",\"file\":\"%t/test.cpp\"}]" | sed -e 's/\\/\//g' > %t/compile_commands.json
// RUN: cp "%s" "%t/test.cpp"
// RUN: touch "%t/clang-check-test.h"
// RUN: not clang-check -p "%t" "%t/test.cpp" 2>&1|FileCheck %s
// FIXME: Make the above easier.

#include "clang-check-test.h"

// CHECK: a type specifier is required
invalid;
