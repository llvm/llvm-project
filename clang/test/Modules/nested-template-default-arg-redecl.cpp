// RUN: rm -rf %t
// RUN: %clang_cc1 -x c++ -fmodules -fimplicit-module-maps -fmodules-cache-path=%t \
// RUN:     -I %S/Inputs/nested-template-default-arg-redecl -std=c++14 \
// RUN:     -fmodules-local-submodule-visibility -w -verify %s

// expected-no-diagnostics

#include "alias2.h"
#include "var2.h"
#include "strct2.h"
#include "func2.h"

auto var = &var_outer::var<>;
auto func = &func_outer::func<>;
strct_outer::strct<> *strct;
alias_outer::alias<> *alias;
