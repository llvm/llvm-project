// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fimplicit-module-maps -I%S/Inputs/missing-header-local-visibility %s
// RUN: %clang_cc1 -fmodules-local-submodule-visibility -fimplicit-module-maps -I%S/Inputs/missing-header-local-visibility -x c++-header %S/Inputs/missing-header-local-visibility/all.h
// RUN: %clang_cc1 -fmodule-name=M -std=c++2a -fimplicit-module-maps -I%S/Inputs/missing-header-local-visibility -x c++-header %s
// RUN: %clang_cc1 -fmodule-name=M -std=c++2a -fimplicit-module-maps -I%S/Inputs/missing-header-local-visibility -x c++-header %S/Inputs/missing-header-local-visibility/all.h

#include "a.h"
#include "b.h"
