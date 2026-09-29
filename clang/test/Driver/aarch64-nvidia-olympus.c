// RUN: %clang --target=aarch64 -mcpu=olympus -### -c %s 2>&1 | FileCheck -check-prefix=olympus %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=olympus -### -c %s 2>&1 | FileCheck -check-prefix=olympus %s
// RUN: %clang --target=aarch64 -mtune=olympus -### -c %s 2>&1 | FileCheck -check-prefix=olympus-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=olympus -### -c %s 2>&1 | FileCheck -check-prefix=olympus-TUNE %s
// olympus: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "olympus"
// olympus-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=olympus -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-olympus %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=olympus -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-olympus %s
// RUN: %clang --target=arm64 -mtune=olympus -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-olympus-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=olympus -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-olympus-TUNE %s
// ARM64-olympus: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "olympus"
// ARM64-olympus-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
