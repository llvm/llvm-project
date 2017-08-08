#include "Inputs/HeaderWithSymbol.h"
#define FOO int bar;
FOO

int foo;

// RUN: not clang-rename -new-name=qux -offset=259 %s -- 2>&1 | FileCheck %s
// CHECK-NOT: CHECK
// CHECK: error: SourceLocation in file {{.*}}InvalidOffset.cpp at offset 259 is invalid
