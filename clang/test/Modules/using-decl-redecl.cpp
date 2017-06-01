// RUN: rm -rf %t
// RUN: %clang_cc1 -fmodules -fmodules-cache-path=%t \
// RUN:            -fmodule-map-file=%S/Inputs/using-decl-redecl/module.modulemap \
// RUN:            -I%S/Inputs/using-decl-redecl \
// RUN:            -Wno-modules-ambiguous-internal-linkage \
// RUN:            -verify %s

#include "d.h"

const int n = 0;
namespace M { using ::n; }

#include "c.h"

N::clstring y = b;

// Use a typo to trigger import of all declarations in N.
N::clstrinh s; // expected-error {{did you mean 'clstring'}}
// expected-note@a.h:3 {{here}}

namespace M { using N::n; }
