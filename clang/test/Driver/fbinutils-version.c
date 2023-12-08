// RUN: %clang -### -c -target x86_64-linux %s -fbinutils-version=none 2>&1 | FileCheck %s --check-prefix=NONE

// NONE: "-fbinutils-version=none"

// RUN: %clang -### -c -target aarch64-linux %s -fbinutils-version=2 2>&1 | FileCheck %s --check-prefix=CHECK2

// CHECK2: "-fbinutils-version=2"

// RUN: %clang -### -c -target aarch64-linux %s -fbinutils-version=2.35 2>&1 | FileCheck %s --check-prefix=CHECK2_35

// CHECK2_35: "-fbinutils-version=2.35"

/// Disallow -fbinutils-version=0 because we use $major==0 to indicate the MC
/// default in the backend.
// RUN: not %clang -c -target x86_64-linux %s -fbinutils-version=0 2>&1 | FileCheck %s --check-prefix=ERR0

// ERR0: error: invalid argument '0' to -fbinutils-version=

// RUN: not %clang -c -target x86_64-linux %s -fbinutils-version=nan 2>&1 | FileCheck %s --check-prefix=ERR1

// ERR1: error: invalid argument 'nan' to -fbinutils-version=

// RUN: not %clang -c -target x86_64-linux %s -fbinutils-version=2. 2>&1 | FileCheck %s --check-prefix=ERR2

// ERR2: error: invalid argument '2.' to -fbinutils-version=

// RUN: not %clang -c -target x86_64-linux %s -fbinutils-version=3.-14 2>&1 | FileCheck %s --check-prefix=ERR3

// ERR3: error: invalid argument '3.-14' to -fbinutils-version=
