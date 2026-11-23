// RUN: %clang --target=aarch64 -mcpu=rigel -### -c %s 2>&1 | FileCheck -check-prefix=rigel %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=rigel -### -c %s 2>&1 | FileCheck -check-prefix=rigel %s
// RUN: %clang --target=aarch64 -mtune=rigel -### -c %s 2>&1 | FileCheck -check-prefix=rigel-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=rigel -### -c %s 2>&1 | FileCheck -check-prefix=rigel-TUNE %s
// rigel: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "rigel"
// rigel-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" {{.*}} "-tune-cpu" "rigel"

// RUN: %clang --target=arm64 -mcpu=rigel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-rigel %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=rigel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-rigel %s
// RUN: %clang --target=arm64 -mtune=rigel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-rigel-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=rigel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-rigel-TUNE %s
// ARM64-rigel: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "rigel"
// ARM64-rigel-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic" {{.*}} "-tune-cpu" "rigel"
