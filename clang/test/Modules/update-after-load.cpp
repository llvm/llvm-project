// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -I %S/Inputs/update-after-load -verify -fmodules-cache-path=%t %s

// expected-no-diagnostics
#include "a.h"
namespace llvm {}
#include "b.h"
void llvm::f() {}
