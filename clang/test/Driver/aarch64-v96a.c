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
