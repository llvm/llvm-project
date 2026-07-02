// RUN: rm -rf %t && split-file %s %t
// RUN: chmod -r %t/no-perm/header.h
// RUN: %clang_cc1 -fsyntax-only -I %t/no-perm -I %t/include %t/tu.c

//--- no-perm/header.h
#error "no permission"
//--- include/header.h
//--- tu.c
#if __has_include("header.h")
#include "header.h"
#endif
