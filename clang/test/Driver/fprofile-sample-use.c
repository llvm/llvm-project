/// GCC -fauto-profile (without =) is rejected.
/// -fprofile-sample-use without = is rejected as well.
// RUN: not %clang -### -S -fauto-profile -fprofile-sample-use %s 2>&1 | FileCheck %s --check-prefix=ERR
// ERR: error: unknown argument: '-fauto-profile'
// ERR: error: unknown argument: '-fprofile-sample-use'
