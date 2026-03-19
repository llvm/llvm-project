// RUN: %clang --target=aarch64 -mcpu=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=fujitsu-monaka %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=fujitsu-monaka %s
// RUN: %clang --target=aarch64 -mtune=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=fujitsu-monaka-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=fujitsu-monaka-TUNE %s
// fujitsu-monaka: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "fujitsu-monaka"
// fujitsu-monaka-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-fujitsu-monaka %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-fujitsu-monaka %s
// RUN: %clang --target=arm64 -mtune=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-fujitsu-monaka-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=fujitsu-monaka -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-fujitsu-monaka-TUNE %s
// ARM64-fujitsu-monaka: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "fujitsu-monaka"
// ARM64-fujitsu-monaka-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
