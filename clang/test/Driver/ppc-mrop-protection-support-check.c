// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr10 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=HASROP
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power10 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=HASROP
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr9 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=HASROP
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power9 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=HASROP
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr8 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=HASROP
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power8 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=HASROP

// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr7 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=NOROP
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power7 -mrop-protection %s 2>&1 | FileCheck %s --check-prefix=NOROP

#ifdef __ROP_PROTECTION__
static_assert(false, "ROP Protection enabled");
#endif

// HASROP: ROP Protection enabled
// HASROP-NOT: option '-mrop-protection' cannot be specified with
// NOROP: option '-mrop-protection' cannot be specified with

