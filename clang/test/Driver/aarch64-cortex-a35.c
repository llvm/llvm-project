// RUN: %clang --target=aarch64 -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-TUNE %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-TUNE %s
// CA35: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a35"
// CA35-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35 %s
// RUN: %clang --target=arm64 -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35-TUNE %s
// ARM64-CA35: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a35"
// ARM64-CA35-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64_be -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE %s
// RUN: %clang --target=aarch64_be -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE-TUNE %s
// CA35-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a35"
// CA35-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
