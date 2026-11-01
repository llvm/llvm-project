// ===== Features supported on aarch64 =====

// FAT features (Future Architecture Technologies)

// RUN: %clang -target aarch64 -march=armv9.7a+mops-go -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-MOPS-GO %s
// RUN: %clang -target aarch64 -march=armv9.7-a+mops-go -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-MOPS-GO %s
// VFAT-MOPS-GO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+mops-go"

// RUN: %clang -target aarch64 -march=armv9.7a+poe2 -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-POE2 %s
// RUN: %clang -target aarch64 -march=armv9.7-a+poe2 -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-POE2 %s
// VFAT-POE2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+poe2"

// RUN: %clang -target aarch64 -march=armv9.7a+tev -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-TEV %s
// RUN: %clang -target aarch64 -march=armv9.7-a+tev -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-TEV %s
// VFAT-TEV: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+tev"

// RUN: %clang -target aarch64 -march=armv9.7a+btie -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-BTIE %s
// RUN: %clang -target aarch64 -march=armv9.7-a+btie -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-BTIE %s
// VFAT-BTIE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+btie"

// RUN: %clang -target aarch64 -march=armv9.8a+sme-f16f8f16 -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-SME-F16F8F16 %s
// RUN: %clang -target aarch64 -march=armv9.8-a+sme-f16f8f16 -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-SME-F16F8F16 %s
// VFAT-SME-F16F8F16: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.8a"{{.*}} "-target-feature" "+sme-f16f8f16"

// RUN: %clang -target aarch64 -march=armv9.8a+fpv2 -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-FPV2 %s
// RUN: %clang -target aarch64 -march=armv9.8-a+fpv2 -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-FPV2 %s
// VFAT-FPV2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.8a"{{.*}} "-target-feature" "+fpv2"
