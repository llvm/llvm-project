// RUN: rm -rf %t
// RUN: mkdir -p %t/Src
// RUN: cp "%s" "%t/Src/test.cpp"
// RUN: mkdir -p %t/Include
// RUN: cp "%S/Inputs/fixed-header.h" "%t/Include/"
// -I flag is relative to %t (where compile_flags is), not Src/.
// RUN: echo '-IInclude/' >> %t/compile_flags.txt
// RUN: echo "  -Dklazz=class   " >> %t/compile_flags.txt
// RUN: echo '-std=c++11' >> %t/compile_flags.txt
// RUN: clang-check "%t/Src/test.cpp" 2>&1
// RUN: echo > %t/compile_flags.txt
// RUN: not clang-check "%t/Src/test.cpp" 2>&1 | FileCheck "%s" -check-prefix=NODB

// NODB: unknown type name 'klazz'
klazz F{};

// NODB: 'fixed-header.h' file not found
#include "fixed-header.h"
static_assert(SECRET_SYMBOL == 1, "");
