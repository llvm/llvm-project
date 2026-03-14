// RUN: %clang --target=aarch64_be -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3 %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3 %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3 %s
// RUN: %clang --target=aarch64_be -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-TUNE %s
// M3: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m3"
// M3-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3 %s
// RUN: %clang --target=arm64 -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3-TUNE %s
// ARM64-M3: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "exynos-m3"
// ARM64-M3-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64_be -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE %s
// RUN: %clang --target=aarch64_be -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE-TUNE %s
// M3-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m3"
// M3-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
