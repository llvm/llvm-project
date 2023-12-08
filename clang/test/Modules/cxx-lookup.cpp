// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t %s -I%S/Inputs/cxx-lookup -verify
// expected-no-diagnostics
namespace llvm {}
#include "c2.h"
llvm::GlobalValue *p;

#include "na.h"
namespace N { struct foo; }
#include "nb.h"
N::foo *use_n_foo;
