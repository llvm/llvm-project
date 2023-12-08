// RUN: %clang --target=aarch64_be -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5 %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5 %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5 %s
// RUN: %clang --target=aarch64_be -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-TUNE %s
// M5: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m5" "-target-feature" "+v8.2a"
// M5-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"
// M5-TUNE-NOT: "+v8.2a"

// RUN: %clang --target=arm64 -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5 %s
// RUN: %clang --target=arm64 -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5-TUNE %s
// ARM64-M5: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "exynos-m5" "-target-feature" "+v8.2a"
// ARM64-M5-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
// ARM64-M5-TUNE-NOT: "+v8.2a"

// RUN: %clang --target=aarch64_be -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE %s
// RUN: %clang --target=aarch64_be -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE-TUNE %s
// M5-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m5" "-target-feature" "+v8.2a"
// M5-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
// M5-BE-TUNE-NOT: "+v8.2a"
