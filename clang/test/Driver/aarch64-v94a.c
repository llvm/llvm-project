// RUN: %clang -target aarch64 -march=armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A %s
// RUN: %clang -target aarch64 -march=armv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A %s
// GENERICV94A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.4a"
// RUN: %clang -target aarch64_be -march=armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV94A-BE %s
// GENERICV94A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.4a"

