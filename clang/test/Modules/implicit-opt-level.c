// This test checks that under implicit modules, different optimization levels
// get different context hashes.

// RUN: rm -rf %t
// RUN: split-file %s %t

//--- module.modulemap
module M { header "M.h" }
//--- M.h
//--- tu.c
#include "M.h"

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -O0 -fsyntax-only %t/tu.c
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t/cache -O1 -fsyntax-only %t/tu.c
// RUN: find %t/cache -name "M-*.pcm" | count 2
