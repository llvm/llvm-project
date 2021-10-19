// RUN: %clang -### -c -target powerpc64 %s -mcpu=ppc64 2>&1 | FileCheck --check-prefix=MCPU_PPC64 %s
// MCPU_PPC64: "-target-cpu" "ppc64"

/// We cannot check much for -mcpu=native, but it should be replaced by a CPU name.
// RUN: %clang -### -c -target powerpc64 %s -mcpu=native 2>&1 | FileCheck --check-prefix=MCPU_NATIVE %s
// MCPU_NATIVE-NOT: "-target-cpu" "native"

// RUN: %clang -### -c -target powerpc64 %s -mcpu=7400 2>&1 | FileCheck --check-prefix=MCPU_7400 %s
// MCPU_7400: "-target-cpu" "7400"

/// The following -mcpu= have their own -target-cpu values.
// RUN: %clang -### -c -target powerpc64 %s -mcpu=G4 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c -target powerpc64 %s -mcpu=7450 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c -target powerpc64 %s -mcpu=G4+ 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c -target powerpc64 %s -mcpu=970 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c -target powerpc64 %s -mcpu=G5 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c -target powerpc64 %s -mcpu=pwr6 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c -target powerpc64 %s -mcpu=pwr7 2>&1 | FileCheck %s --check-prefix=NO_PPC64
// RUN: %clang -### -c -target powerpc64 %s -mcpu=pwr8 2>&1 | FileCheck %s --check-prefix=NO_PPC64

// NO_PPC64-NOT: "-target-cpu" "ppc64"
