// RUN: %clang -target sparcv9 -### -c %s 2>&1 | FileCheck -check-prefix=SPARCV9 %s
// SPARCV9: "-cc1"{{.*}} "-triple" "sparcv9"

// RUN: %clang -target sparc64 -### -c %s 2>&1 | FileCheck -check-prefix=SPARC64 %s
// SPARC64: "-cc1"{{.*}} "-triple" "sparc64"

// RUN: %clang -target sparcv9 -mcpu=v9 -### -c %s 2>&1 | FileCheck -check-prefix=SPARCV9_V9 %s
// SPARCV9_V9: "-cc1"{{.*}} "-triple" "sparcv9"{{.*}} "-target-cpu" "v9"

// RUN: %clang -target sparcv9 -mcpu=ultrasparc -### -c %s 2>&1 | FileCheck -check-prefix=SPARCV9_US %s
// SPARCV9_US: "-cc1"{{.*}} "-triple" "sparcv9"{{.*}} "-target-cpu" "ultrasparc"

// RUN: %clang -target sparc -### -c %s 2>&1 | FileCheck -check-prefix=SPARC %s
// SPARC: "-cc1"{{.*}} "-triple" "sparc"

// RUN: %clang -target sparc -mcpu=v9 -### -c %s 2>&1 | FileCheck -check-prefix=SPARC_V9 %s
// SPARC_V9: "-cc1"{{.*}} "-triple" "sparc"{{.*}} "-target-cpu" "v9"

// RUN: %clang -target sparc -mcpu=ultrasparc -### -c %s 2>&1 | FileCheck -check-prefix=SPARC_US %s
// SPARC_US: "-cc1"{{.*}} "-triple" "sparc"{{.*}} "-target-cpu" "ultrasparc"

