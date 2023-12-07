// SVE2 is enabled by default on Armv9-A but it can be disabled
// RUN: %clang --target=aarch64 -march=armv9a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang --target=aarch64 -march=armv9-a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9-a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9-a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// GENERICV9A-NOSVE2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9a" "-target-feature" "+sve" "-target-feature" "-sve2"
