// RUN: %clang --target=aarch64 -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64 -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// GENERICV9A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9a"{{.*}} "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64_be -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64_be -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// GENERICV9A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9a"{{.*}} "-target-feature" "+sve" "-target-feature" "+sve2"
