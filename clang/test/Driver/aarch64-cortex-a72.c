// RUN: %clang --target=aarch64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-TUNE %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-TUNE %s
// CA72: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a72"
// CA72-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// RUN: %clang --target=arm64 -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72-TUNE %s
// ARM64-CA72: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a72"
// ARM64-CA72-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64_be -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang --target=aarch64_be -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE-TUNE %s
// CA72-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a72"
// CA72-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64 -mcpu=cortex-a72 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-A72 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a53 -mcpu=cortex-a72  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-A72 %s
// MCPU-MTUNE-A72: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a72"
