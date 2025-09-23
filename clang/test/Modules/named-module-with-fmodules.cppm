// Checks that Clang modules can be imported from within the global module 
// fragment of a named module interface unit.
// Fixes #159768.

// RUN: rm -rf %t
// RUN: mkdir -p %t
// RUN: split-file %s %t

// RUN: %clang -std=c++23 -fmodules -fmodule-map-file=%t/module.modulemap \
// RUN:   -fmodules-cache-path=%t --precompile %t/A.cppm -o %t/A.pcm

//--- module.modulemap
module foo { header "foo.h" }

//--- foo.h
// empty

//--- A.cppm
module;
#include "foo.h"
export module A;

export auto A() -> int { return 42; }

