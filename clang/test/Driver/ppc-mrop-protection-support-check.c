// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr10 -mrop-protect %s 2>&1 | FileCheck %s --check-prefix=HASROP
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power8 -mrop-protect %s 2>&1 | FileCheck %s --check-prefix=HASROP

// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=pwr7 -mrop-protect %s 2>&1 | FileCheck %s --check-prefix=NOROP

// RUN: not %clang -target powerpc64-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power8 -mrop-protect %s 2>&1 | FileCheck %s --check-prefix=ELFV1
// RUN: not %clang -target powerpc64le-unknown-linux-gnu -fsyntax-only \
// RUN:   -mcpu=power8 -mrop-protect -mabi=elfv1 %s 2>&1 | \
// RUN:   FileCheck %s --check-prefix=ELFV1

// RUN: not %clang -target powerpc-unknown-linux -fsyntax-only \
// RUN: -mcpu=pwr8 -mrop-protect %s 2>&1 | FileCheck %s --check-prefix=32BIT
// RUN: not %clang -target powerpc-unknown-aix -fsyntax-only \
// RUN: -mcpu=pwr8 -mrop-protect %s 2>&1 | FileCheck %s --check-prefix=32BIT

#ifdef __ROP_PROTECT__
#if defined(__CALL_ELF) && __CALL_ELF == 1
#error "ROP protection not supported with 64-bit elfv1 abi"
#endif
#endif

#ifdef __ROP_PROTECT__
static_assert(false, "ROP Protect enabled");
#endif

// HASROP: ROP Protect enabled
// HASROP-NOT: option '-mrop-protect' cannot be specified with
// NOROP: option '-mrop-protect' cannot be specified with

// 32BIT: option '-mrop-protect' cannot be specified on this target
// ELFV1: '-mrop-protect' can only be used with the 'elfv2' ABI
