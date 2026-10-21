// RUN: %clang -### -fpseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=YESPROBE
// RUN: %clang -### -fno-pseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=NOPROBE
// RUN: %clang -### -fpseudo-probe-for-profiling -fdebug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=YESPROBE --check-prefix=YESDEBUG  
// RUN: %clang -### -fpseudo-probe-for-profiling -funique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=YESPROBE
// RUN: %clang -### -fpseudo-probe-for-profiling -fno-unique-internal-linkage-names %s 2>&1 | FileCheck %s --check-prefix=NONAME

// YESDEBUG: -fdebug-info-for-profiling
// YESPROBE: -fpseudo-probe-for-profiling
// YESPROBE: -funique-internal-linkage-names
// NOPROBE-NOT: -fpseudo-probe-for-profiling
// NOPROBE-NOT: -funique-internal-linkage-names
// NONAME: -fpseudo-probe-for-profiling
// NONAME-NOT: -funique-internal-linkage-names

// On Darwin, -fpseudo-probe-for-profiling should trigger dsymutil
// RUN: %clang -target arm64-apple-darwin -### -o foo -fpseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=CHECK-DSYMUTIL-PSEUDO-PROBE
// CHECK-DSYMUTIL-PSEUDO-PROBE: "-cc1"
// CHECK-DSYMUTIL-PSEUDO-PROBE: ld
// CHECK-DSYMUTIL-PSEUDO-PROBE: dsymutil

// RUN: %clang -target arm64-apple-darwin -### -o foo -fno-pseudo-probe-for-profiling %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-DSYMUTIL-PSEUDO-PROBE
// CHECK-NO-DSYMUTIL-PSEUDO-PROBE: "-cc1"
// CHECK-NO-DSYMUTIL-PSEUDO-PROBE: ld
// CHECK-NO-DSYMUTIL-PSEUDO-PROBE-NOT: dsymutil

// On Darwin, -fdebug-info-for-profiling should trigger dsymutil
// RUN: %clang -target arm64-apple-darwin -### -o foo -fdebug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=CHECK-DSYMUTIL-DEBUG-PROF
// CHECK-DSYMUTIL-DEBUG-PROF: "-cc1"
// CHECK-DSYMUTIL-DEBUG-PROF: ld
// CHECK-DSYMUTIL-DEBUG-PROF: dsymutil

// RUN: %clang -target arm64-apple-darwin -### -o foo -fno-debug-info-for-profiling %s 2>&1 | FileCheck %s --check-prefix=CHECK-NO-DSYMUTIL-DEBUG-PROF
// CHECK-NO-DSYMUTIL-DEBUG-PROF: "-cc1"
// CHECK-NO-DSYMUTIL-DEBUG-PROF: ld
// CHECK-NO-DSYMUTIL-DEBUG-PROF-NOT: dsymutil
