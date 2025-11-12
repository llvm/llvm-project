// ===== Features supported on aarch64 =====

// FAT features (Future Architecture Technologies)

// RUN: %clang -target aarch64 -march=armv9.7a+mops-go -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-MOPS-GO %s
// RUN: %clang -target aarch64 -march=armv9.7-a+mops-go -### -c %s 2>&1 | FileCheck -check-prefix=VFAT-MOPS-GO %s
// VFAT-MOPS-GO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+mops-go"
