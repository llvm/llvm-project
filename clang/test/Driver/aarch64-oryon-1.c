// RUN: %clang -target aarch64 -mcpu=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=Phoenix %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=Phoenix %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=Phoenix %s
// RUN: %clang -target aarch64 -mtune=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=Phoenix-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=Phoenix-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=Phoenix-TUNE %s
// Phoenix: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "oryon-1" "-target-feature" "+v8.6a"
// Phoenix-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-Phoenix %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-Phoenix %s
// RUN: %clang -target arm64 -mtune=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-Phoenix-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=oryon-1 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-Phoenix-TUNE %s
// ARM64-Phoenix: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "oryon-1" "-target-feature" "+v8.6a"
// ARM64-Phoenix-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=oryon-1 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-Phoenix %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=oryon-1  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-Phoenix %s
// MCPU-MTUNE-Phoenix: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "oryon-1"
