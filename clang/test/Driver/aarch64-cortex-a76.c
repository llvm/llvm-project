// RUN: %clang --target=aarch64 -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76-TUNE %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76-TUNE %s
// CORTEX-A76: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a76"
// CORTEX-A76-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76 %s
// RUN: %clang --target=arm64 -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76-TUNE %s
// ARM64-CORTEX-A76: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a76"
// ARM64-CORTEX-A76-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
