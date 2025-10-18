// ===== Base v9.7a architecture =====

// RUN: %clang -target aarch64 -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// GENERICV97A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}}

// RUN: %clang -target aarch64_be -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// GENERICV97A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}}

// ===== Features supported on aarch64 =====
