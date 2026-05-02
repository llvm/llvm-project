// This checks that implicitly-built modules produce identical PCM
// files regardless of the spelling of the same module cache path.

// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only %t/tu.c \
// RUN:   -fmodules-cache-path=%t/cache -fdisable-module-hash
// RUN: mv %t/cache/M.pcm %t/M.pcm
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fsyntax-only %t/tu.c \
// RUN:   -fmodules-cache-path=%t/./cache -fdisable-module-hash
// RUN: diff %t/./cache/M.pcm %t/M.pcm

//--- tu.c
#include "M.h"
//--- M.h
//--- module.modulemap
module M { header "M.h" }
