// RUN: rm -rf %t
// RUN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps %s -fmodules-cache-path=%t -verify -I%S/Inputs/macro-masking
// RxN: %clang_cc1 -fsyntax-only -fmodules -fimplicit-module-maps -fmodules-local-submodule-visibility %s -fmodules-cache-path=%t -verify -I%S/Inputs/macro-masking -DLOCAL_VISIBILITY
// expected-no-diagnostics

#include "a.h"

#ifdef LOCAL_VISIBILITY
# ifndef MACRO
#  error should still be defined, undef does not override define
# endif
#else
# ifdef MACRO
#  error should have been undefined!
# endif
#endif
