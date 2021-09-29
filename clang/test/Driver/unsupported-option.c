// RUN: not %clang %s --hedonism -### 2>&1 | \
// RUN: FileCheck %s
// CHECK: error: unsupported option '--hedonism'

// RUN: not %clang %s --hell -### 2>&1 | \
// RUN: FileCheck %s --check-prefix=DID-YOU-MEAN
// DID-YOU-MEAN: error: unsupported option '--hell'; did you mean '--help'?

// RUN: not %clang -fprofile-instr-generate --target=powerpc-ibm-aix %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=INVALID-AIX-PROFILE
// INVALID-AIX-PROFILE: error: unsupported option '-fprofile-instr-generate' for target

// RUN: not %clang -fprofile-sample-use=code.prof --target=powerpc-ibm-aix %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=AIX-PROFILE-SAMPLE
// AIX-PROFILE-SAMPLE: error: unsupported option '-fprofile-sample-use=' for target

// RUN: not %clang --target=powerpc-ibm-aix %s -mlong-double-128 2>&1 | \
// RUN: FileCheck %s --check-prefix=AIX-LONGDOUBLE128-ERR
// AIX-LONGDOUBLE128-ERR: error: unsupported option '-mlong-double-128' for target 'powerpc-ibm-aix'

// RUN: not %clang --target=powerpc64-ibm-aix %s -mlong-double-128 2>&1 | \
// RUN: FileCheck %s --check-prefix=AIX64-LONGDOUBLE128-ERR
// AIX64-LONGDOUBLE128-ERR: error: unsupported option '-mlong-double-128' for target 'powerpc64-ibm-aix'
