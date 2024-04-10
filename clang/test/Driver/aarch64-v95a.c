// ===== Base v9.5a architecture =====

// RUN: %clang -target aarch64 -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64 -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// GENERICV95A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.5a"

// RUN: %clang -target aarch64_be -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// GENERICV95A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.5a"

// ===== Features supported on aarch64 =====

// RUN: %clang -target aarch64 -march=armv9.5a+cpa -### -c %s 2>&1 | FileCheck -check-prefix=V95A-CPA %s
// RUN: %clang -target aarch64 -march=armv9.5-a+cpa -### -c %s 2>&1 | FileCheck -check-prefix=V95A-CPA %s
// V95A-CPA: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.5a"{{.*}} "-target-feature" "+cpa"

// RUN: %clang -target aarch64 -march=armv9.5a+pauth-lr -### -c %s 2>&1 | FileCheck -check-prefix=V95A-PAUTHLR %s
// RUN: %clang -target aarch64 -march=armv9.5-a+pauth-lr -### -c %s 2>&1 | FileCheck -check-prefix=V95A-PAUTHLR %s
// V95A-PAUTHLR: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.5a"{{.*}} "-target-feature" "+pauth-lr"

// RUN: %clang -target aarch64 -march=armv9.5a+tlbiw -### -c %s 2>&1 | FileCheck -check-prefix=V95A-TLBIW %s
// RUN: %clang -target aarch64 -march=armv9.5-a+tlbiw -### -c %s 2>&1 | FileCheck -check-prefix=V95A-TLBIW %s
// V95A-TLBIW: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.5a"{{.*}} "-target-feature" "+tlbiw"
