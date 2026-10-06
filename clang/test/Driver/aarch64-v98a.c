// ===== Base v9.8a architecture =====

// RUN: %clang -target aarch64 -march=armv9.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A %s
// RUN: %clang -target aarch64 -march=armv9.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A %s
// GENERICV98A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.8a"{{.*}} "-target-feature" "+fprcvt"{{.*}} "-target-feature" "+sve2p3"{{.*}}

// RUN: %clang -target aarch64_be -march=armv9.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV98A-BE %s
// GENERICV98A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.8a"{{.*}} "-target-feature" "+fprcvt"{{.*}} "-target-feature" "+sve2p3"{{.*}}

// ===== Features supported on aarch64 =====
//
// RUN: %clang -target aarch64 -march=armv9.8a+cflt -### -c %s 2>&1 | FileCheck -check-prefix=V98A-CFLT %s
// RUN: %clang -target aarch64 -march=armv9.8-a+cflt -### -c %s 2>&1 | FileCheck -check-prefix=V98A-CFLT %s
// V98A-CFLT: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.8a"{{.*}} "-target-feature" "+cflt"
