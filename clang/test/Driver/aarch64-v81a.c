// RUN: %clang --target=aarch64_be -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// GENERICV81A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"{{.*}} "-target-feature" "+v8.1a"{{.*}} "-target-feature" "+neon"

// RUN: %clang --target=aarch64 -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang --target=aarch64 -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// GENERICV81A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+v8.1a"{{.*}} "-target-feature" "+neon"{{.*}}

// RUN: %clang --target=arm64 -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// RUN: %clang --target=arm64 -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// RUN: %clang --target=arm64 -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// RUN: %clang --target=arm64 -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// ARM64-GENERICV81A: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"{{.*}} "-target-feature" "+v8.1a"{{.*}} "-target-feature" "+neon"

// ===== Architecture extensions =====

// RUN: %if aarch64-registered-target %{ %clang -target aarch64 -march=armv8.1-a --print-enabled-extensions | FileCheck -check-prefix=ARCH-EXTENSION --implicit-check-not FEAT_ %s %}
// ARCH-EXTENSION: FEAT_AdvSIMD
// ARCH-EXTENSION: FEAT_CRC32
// ARCH-EXTENSION: FEAT_ETE
// ARCH-EXTENSION: FEAT_FP
// ARCH-EXTENSION: FEAT_LOR
// ARCH-EXTENSION: FEAT_LSE
// ARCH-EXTENSION: FEAT_PAN
// ARCH-EXTENSION: FEAT_RDM
// ARCH-EXTENSION: FEAT_TRBE
// ARCH-EXTENSION: FEAT_VHE
