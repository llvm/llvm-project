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

// RUN: not %clang -fprofile-generate --target=powerpc-ibm-aix %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=AIX-PROFILE-LTO
// AIX-PROFILE-LTO: error: invalid argument '-fprofile-generate' only allowed with '-flto'

// RUN: not %clang -fprofile-generate -flto=thin --target=powerpc64-ibm-aix %s 2>&1 | \
// RUN: FileCheck %s --check-prefix=AIX-PROFILE-THINLTO
// AIX-PROFILE-THINLTO: error: invalid argument '-fprofile-generate' only allowed with '-flto'
