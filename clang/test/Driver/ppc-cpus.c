// RUN: %clang -### -c --target=powerpc64 %s -mcpu=ppc64 2>&1 | FileCheck --check-prefix=MCPU_PPC64 %s
// MCPU_PPC64: "-target-cpu" "ppc64"

/// We cannot check much for -mcpu=native, but it should be replaced by a CPU name.
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=native 2>&1 | FileCheck --check-prefix=MCPU_NATIVE %s
// MCPU_NATIVE-NOT: "-target-cpu" "native"

/// Check that we are passing unknown mcpu options to the backend so an error
/// can be triggered.
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=asdf1234 2>&1 | FileCheck --check-prefix=MCPU_UNKNOWN %s
// MCPU_UNKNOWN: "-target-cpu" "asdf1234"

/// Check for the unknown target error if an unknown mcpu option is used.
// RUN: not %clang -c --target=powerpc64 %s -mcpu=asdf1234 2>&1 | FileCheck --check-prefix=MCPU_ERR %s
// MCPU_ERR: unknown target CPU 'asdf1234'

// RUN: %clang -### -c --target=powerpc64 %s -mcpu=7400 2>&1 | FileCheck --check-prefix=MCPU_7400 %s
// MCPU_7400: "-target-cpu" "7400"

/// The following -mcpu= have their own -target-cpu values.
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=G4 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=7450 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=G4+ 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=970 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=G5 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=pwr6 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=pwr7 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=pwr8 2>&1 | FileCheck %s --check-prefix=NO_PPC64

// NO_PPC64-NOT: "-target-cpu" "ppc64"

// RUN: %clang -### -c --target=powerpc64 %s -mcpu=generic -mtune=pwr9 2>&1 | FileCheck %s --check-prefix=TUNE
// TUNE: "-target-cpu" "ppc64" "-tune-cpu" "pwr9"
// RUN: %clang -### -c --target=powerpc64le %s -mcpu=power9 -mtune=power10 2>&1 | FileCheck %s --check-prefix=TUNE-LONG
// TUNE-LONG: "-target-cpu" "pwr9" "-tune-cpu" "pwr10"

/// Test mcpu options that are equivalent to "generic"
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=generic 2>&1 | FileCheck %s --check-prefix=GENERIC
// RUN: %clang -### -c --target=powerpc64 %s -mcpu=405     2>&1 | FileCheck %s --check-prefix=GENERIC
//
// GENERIC: "-target-cpu" "ppc64"

// RUN: %clang -### -c --target=powerpc64 %s -march=generic 2>&1 | FileCheck --check-prefix=MARCH %s
// MARCH: error: unsupported option '-march=' for target 'powerpc64'
