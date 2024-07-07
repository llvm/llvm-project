// RUN: %clang --target=aarch64 -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64 -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// GENERICV83A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+v8.3a"{{.*}} "-target-feature" "+neon"

// RUN: %clang --target=aarch64_be -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// GENERICV83A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v8.3a"{{.*}} "-target-feature" "+neon"

// ===== Architecture extensions =====

// RUN: %if aarch64-registered-target %{ %clang -target aarch64 -march=armv8.3-a --print-enabled-extensions | FileCheck -check-prefix=ARCH-EXTENSION --implicit-check-not FEAT_ %s %}
// ARCH-EXTENSION: FEAT_AdvSIMD
// ARCH-EXTENSION: FEAT_CCIDX
// ARCH-EXTENSION: FEAT_CRC32
// ARCH-EXTENSION: FEAT_DPB
// ARCH-EXTENSION: FEAT_ETE
// ARCH-EXTENSION: FEAT_FCMA
// ARCH-EXTENSION: FEAT_FP
// ARCH-EXTENSION: FEAT_JSCVT
// ARCH-EXTENSION: FEAT_LOR
// ARCH-EXTENSION: FEAT_LRCPC
// ARCH-EXTENSION: FEAT_LSE
// ARCH-EXTENSION: FEAT_PAN
// ARCH-EXTENSION: FEAT_PAN2
// ARCH-EXTENSION: FEAT_PAuth
// ARCH-EXTENSION: FEAT_RAS, FEAT_RASv1p1
// ARCH-EXTENSION: FEAT_RDM
// ARCH-EXTENSION: FEAT_TRBE
// ARCH-EXTENSION: FEAT_UAO
// ARCH-EXTENSION: FEAT_VHE
