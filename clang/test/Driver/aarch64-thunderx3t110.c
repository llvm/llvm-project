// RUN: %clang --target=aarch64 -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110 %s
// RUN: %clang --target=aarch64 -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-TUNE %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-TUNE %s
// THUNDERX3T110: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "thunderx3t110" "-target-feature" "+v8.3a"
// THUNDERX3T110-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110 %s
// RUN: %clang --target=arm64 -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110-TUNE %s
// ARM64-THUNDERX3T110: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "thunderx3t110" "-target-feature" "+v8.3a"
// ARM64-THUNDERX3T110-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64_be -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE %s
// RUN: %clang --target=aarch64_be -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE-TUNE %s
// THUNDERX3T110-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "thunderx3t110"
// THUNDERX3T110-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64 -mcpu=thunderx3t110 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-THUNDERX3T110 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a53 -mcpu=thunderx3t110  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-THUNDERX3T110 %s
// MCPU-MTUNE-THUNDERX3T110: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "thunderx3t110"
