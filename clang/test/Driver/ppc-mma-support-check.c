// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr10 -mmma %s 2>&1 | FileCheck %s --check-prefix=HASMMA
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power10 -mmma %s 2>&1 | FileCheck %s --check-prefix=HASMMA

// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr9 -mmma %s 2>&1 | FileCheck %s --check-prefix=NOMMA
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr8 -mmma %s 2>&1 | FileCheck %s --check-prefix=NOMMA
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr7 -mmma %s 2>&1 | FileCheck %s --check-prefix=NOMMA
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mmma %s 2>&1 | FileCheck %s --check-prefix=NOMMA

#ifdef __MMA__
static_assert(false, "MMA enabled");
#endif

// HASMMA: MMA enabled
// HASMMA-NOT: option '-mmma' cannot be specified with
// NOMMA: option '-mmma' cannot be specified with

