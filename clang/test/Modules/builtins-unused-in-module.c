// Builtins are only stored in a module if it uses them. Check that importers
// can use builtins that the module doesn't know about, and builtins that the
// module uses. Identifiers from the module are deserialized differently in C
// and C++.

// RUN: rm -rf %t
// RUN: split-file %s %t
// RUN: %clang_cc1 -triple arm64-apple-macosx -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I %t -fsyntax-only -verify %t/tu.c
// RUN: %clang_cc1 -triple arm64-apple-macosx -fmodules -fimplicit-module-maps \
// RUN:   -fmodules-cache-path=%t/cache -I %t -fsyntax-only -verify \
// RUN:   -x c++ %t/tu.c

//--- module.modulemap
module M { header "M.h" }

//--- M.h
static inline int m_popcount(unsigned x) { return __builtin_popcount(x); }
#define M_CTZ(x) __builtin_ctz(x)

//--- tu.c
// expected-no-diagnostics
#include "M.h"
_Static_assert(__builtin_popcount(7) == 3, "");
_Static_assert(M_CTZ(8) == 3, "");
_Static_assert(__builtin_clz(1) == 31, "");
#if !__has_builtin(__builtin_arm_rbit)
#error "missing target builtin"
#endif
