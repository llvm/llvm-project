// RUN: %clang --target=aarch64 -mcpu=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=HIP12 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=HIP12 %s
// RUN: %clang --target=aarch64 -mtune=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=HIP12-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=HIP12-TUNE %s
// HIP12: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "hip12"
// HIP12-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-HIP12 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-HIP12 %s
// RUN: %clang --target=arm64 -mtune=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-HIP12-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=hip12 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-HIP12-TUNE %s
// ARM64-HIP12: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "hip12"
// ARM64-HIP12-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
