// RUN: %clang -### -fpseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=YESPROBE
// RUN: %clang -### -fno-pseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=NOPROBE
// RUN: %clang -### -fpseudo-probe-for-profiling -fdebug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=CONFLICT
// RUN: %clang -### -fpseudo-probe-for-profiling -funique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=YESPROBE
// RUN: %clang -### -fpseudo-probe-for-profiling -fno-unique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=NONAME

// YESPROBE: -fpseudo-probe-for-profiling
// YESPROBE: -funique-internal-linkage-names
// NOPROBE-NOT: -fpseudo-probe-for-profiling
// NOPROBE-NOT: -funique-internal-linkage-names
// NONAME: -fpseudo-probe-for-profiling
// NONAME-NOT: -funique-internal-linkage-names
// CONFLICT: invalid argument
