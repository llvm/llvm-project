
// Check that all the following options print a warning when given a non-existent
// value. But only one warning.

// RUN: not %clangxx -stdlib=nostdlib %s 2>&1 | FileCheck --check-prefix=STDLIB %s
// STDLIB: error: invalid library name in argument '-stdlib=nostdlib'
// STDLIB-EMPTY:
//
// RUN: not %clangxx -rtlib=nortlib %s 2>&1 | FileCheck --check-prefix=RTLIB %s
// RTLIB: error: invalid runtime library name in argument '-rtlib=nortlib'
// RTLIB-EMPTY:
//
// RUN: not %clangxx -unwindlib=nounwindlib %s 2>&1 | FileCheck --check-prefix=UNWINDLIB %s
// UNWINDLIB: error: invalid unwind library name in argument '-unwindlib=nounwindlib'
// UNWINDLIB-EMPTY:
