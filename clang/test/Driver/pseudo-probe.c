// RUN: %clang -### -fpseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=YESPROBE
// RUN: %clang -### -fno-pseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=NOPROBE
// RUN: %clang -### -fpseudo-probe-for-profiling -fdebug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=CONFLICT

// YESPROBE: -fpseudo-probe-for-profiling
// NOPROBE-NOT: -fpseudo-probe-for-profiling
// CONFLICT: invalid argument
