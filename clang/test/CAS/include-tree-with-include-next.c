// RUN: rm -rf %t
// RUN: split-file %s %t

// Normal compilation for baseline.
// RUN: %clang_cc1 %t/t.c -I %t/inc -I %t/inc2 -fsyntax-only -Werror

// RUN: %clang -cc1depscan -o %t/tu.rsp -fdepscan=inline -cc1-args \
// RUN:     -cc1 -fsyntax-only -Werror -fcas-path %t/cas %t/t.c -I %t/inc -I %t/inc2 -Werror
// RUN: %clang @%t/tu.rsp

//--- t.c
#include "t.h"

//--- inc/t.h
#include_next "t.h"

//--- inc2/t.h
