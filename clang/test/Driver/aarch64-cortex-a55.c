// RUN: %clang --target=aarch64 -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-TUNE %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-TUNE %s
// CA55: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a55"
// CA55-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55 %s
// RUN: %clang --target=arm64 -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55-TUNE %s
// ARM64-CA55: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a55"
// ARM64-CA55-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64_be -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE %s
// RUN: %clang --target=aarch64_be -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE-TUNE %s
// CA55-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a55"
// CA55-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
