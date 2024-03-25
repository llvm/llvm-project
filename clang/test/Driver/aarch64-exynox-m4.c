// RUN: %clang --target=aarch64_be -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4 %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4 %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4 %s
// RUN: %clang --target=aarch64_be -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-TUNE %s
// M4: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m4" "-target-feature" "+v8.2a"
// M4-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"
// M4-TUNE-NOT: "+v8.2a"

// RUN: %clang --target=arm64 -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4 %s
// RUN: %clang --target=arm64 -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4-TUNE %s
// ARM64-M4: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "exynos-m4" "-target-feature" "+v8.2a"
// ARM64-M4-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
// ARM64-M4-TUNE-NOT: "+v8.2a"

// RUN: %clang --target=aarch64_be -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE %s
// RUN: %clang --target=aarch64_be -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE-TUNE %s
// M4-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m4" "-target-feature" "+v8.2a"
// M4-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
// M4-BE-TUNE-NOT: "+v8.2a"
