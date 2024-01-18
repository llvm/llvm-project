// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module Mod { header "mod.h" }
//--- mod.h
#if __has_include("textual.h")
#endif
//--- textual.h

//--- tu.c
#include "mod.h"

// RUN: %clang -fsyntax-only %t/tu.c -fmodules -fmodules-cache-path=%t/cache -Werror=non-modular-include-in-module
