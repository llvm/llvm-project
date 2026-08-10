// RUN: %clang --target=aarch64 -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX %s
// RUN: %clang --target=aarch64 -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX-TUNE %s
// A64FX: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "a64fx"
// A64FX-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX %s
// RUN: %clang --target=arm64 -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX-TUNE %s
// ARM64-A64FX: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "a64fx"
// ARM64-A64FX-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
