// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-lifetime-extended-temporary -verify -std=c++11 %s -DORDER=1
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -x c++ -I%S/Inputs/merge-lifetime-extended-temporary -verify -std=c++11 %s -DORDER=2

// expected-no-diagnostics
#if ORDER == 1
#include "c.h"
#include "b.h"
#else
#include "b.h"
#include "c.h"
#endif

static_assert(PtrTemp1 == &LETemp, "");
static_assert(PtrTemp1 == PtrTemp2, "");
