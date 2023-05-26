// RUN: %clang --target=aarch64 -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL %s
// RUN: %clang --target=aarch64 -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL-TUNE %s
// CARMEL: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "carmel"
// CARMEL-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL %s
// RUN: %clang --target=arm64 -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL-TUNE %s
// ARM64-CARMEL: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "carmel"
// ARM64-CARMEL-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
