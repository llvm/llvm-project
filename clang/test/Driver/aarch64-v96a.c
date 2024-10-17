// ===== Base v9.6a architecture =====

// RUN: %clang -target aarch64 -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64 -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// GENERICV96A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"

// RUN: %clang -target aarch64_be -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// GENERICV96A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"
//
// ===== Features supported on aarch64 =====
//
// RUN: %clang -target aarch64 -march=armv9.6a+lsui -### -c %s 2>&1 | FileCheck -check-prefix=V96A-LSUI %s
// RUN: %clang -target aarch64 -march=armv9.6-a+lsui -### -c %s 2>&1 | FileCheck -check-prefix=V96A-LSUI %s
// V96A-LSUI: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+lsui"
//
// RUN: %clang -target aarch64 -march=armv9.6a+occmo -### -c %s 2>&1 | FileCheck -check-prefix=V96A-OCCMO %s
// RUN: %clang -target aarch64 -march=armv9.6-a+occmo -### -c %s 2>&1 | FileCheck -check-prefix=V96A-OCCMO %s
// V96A-OCCMO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+occmo"
//
// RUN: %clang -target aarch64 -march=armv9.6a+pcdphint -### -c %s 2>&1 | FileCheck -check-prefix=V96A-PCDPHINT %s
// RUN: %clang -target aarch64 -march=armv9.6-a+pcdphint -### -c %s 2>&1 | FileCheck -check-prefix=V96A-PCDPHINT %s
// V96A-PCDPHINT: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+pcdphint"
//
