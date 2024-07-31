// RUN: %clang --target=aarch64 -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang --target=aarch64 -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// GENERICV92A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.2a"{{.*}} "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64_be -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang --target=aarch64_be -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// GENERICV92A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.2a"{{.*}} "-target-feature" "+sve" "-target-feature" "+sve2"
