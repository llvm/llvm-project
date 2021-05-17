// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr10 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=HASPRIV
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power10 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=HASPRIV
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr9 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=HASPRIV
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power9 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=HASPRIV
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr8 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=HASPRIV
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power8 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=HASPRIV

// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr7 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=NOPRIV
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power7 -mprivileged %s 2>&1 | FileCheck %s --check-prefix=NOPRIV

#ifdef __PRIVILEGED__
static_assert(false, "Privileged instructions enabled");
#endif

// HASPRIV: Privileged instructions enabled
// HASPRIV-NOT: option '-mprivileged' cannot be specified with
// NOPRIV: option '-mprivileged' cannot be specified with

