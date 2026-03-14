// RUN: %clang --target=aarch64 -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34 %s
// RUN: %clang --target=aarch64 -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-TUNE %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-TUNE %s
// CA34: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a34"
// CA34-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34 %s
// RUN: %clang --target=arm64 -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34-TUNE %s
// ARM64-CA34: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a34"
// ARM64-CA34-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64_be -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE %s
// RUN: %clang --target=aarch64_be -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE-TUNE %s
// RUN: %clang --target=aarch64 -mbig-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE-TUNE %s
// RUN: %clang --target=aarch64_be -mbig-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE-TUNE %s
// CA34-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a34"
// CA34-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
