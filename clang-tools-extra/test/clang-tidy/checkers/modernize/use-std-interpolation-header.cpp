// RUN: %check_clang_tidy -std=c++20-or-later -check-header %S/Inputs/use-std-interpolation.h %s modernize-use-std-interpolation %t -- -header-filter=.* -format-style=llvm

#include "use-std-interpolation.h"
#include "use-std-interpolation.h"
