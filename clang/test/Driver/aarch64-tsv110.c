// RUN: %clang --target=aarch64 -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110 %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110 %s
// RUN: %clang --target=aarch64 -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110-TUNE %s
// RUN: %clang --target=aarch64 -mlittle-endian -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110-TUNE %s
// TSV110: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "tsv110"
// TSV110-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110 %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110 %s
// RUN: %clang --target=arm64 -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110-TUNE %s
// RUN: %clang --target=arm64 -mlittle-endian -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110-TUNE %s
// ARM64-TSV110: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "tsv110"
// ARM64-TSV110-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
