// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fimplicit-module-maps -fmodules-cache-path=%t -I%S/Inputs/merge-decl-order -verify %s
// expected-no-diagnostics

// Check that we include all decls from 'a' before the decls from 'b' in foo's
// redecl chain. If we don't, then name lookup only finds invisible friend
// declarations and the lookup below will fail.
#include "b.h"
N::foo *use;
