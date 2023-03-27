// RUN: %clang --target=aarch64 -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75-TUNE %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75-TUNE %s
// CORTEX-A75: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a75"
// CORTEX-A75-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75 %s
// RUN: %clang --target=arm64 -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75-TUNE %s
// ARM64-CORTEX-A75: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a75"
// ARM64-CORTEX-A75-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
