// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/invalidate-identifiers -emit-llvm-only %s

#include "b.h"
