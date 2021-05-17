// Check that all the following options print a warning when given a
// non-existent value. But only one warning.

// RUN: not %clangxx --target=i386-unknown-linux -stdlib=nostdlib %s 2>&1 | FileCheck --check-prefix=STDLIB %s
// STDLIB: error: invalid library name in argument '-stdlib=nostdlib'
// STDLIB-EMPTY:

// RUN: not %clangxx --target=i386-unknown-linux -rtlib=nortlib %s 2>&1 | FileCheck --check-prefix=RTLIB %s
// RTLIB: error: invalid runtime library name in argument '-rtlib=nortlib'
// RTLIB-EMPTY:

// RUN: not %clangxx --target=i386-unknown-linux -unwindlib=nounwindlib %s 2>&1 | FileCheck --check-prefix=UNWINDLIB %s
// UNWINDLIB: error: invalid unwind library name in argument '-unwindlib=nounwindlib'
// UNWINDLIB-EMPTY:
