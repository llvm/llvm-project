// RUN: %clang --target=aarch64 -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang --target=aarch64 -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// GENERICV91A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.1a"{{.*}} "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64_be -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang --target=aarch64_be -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// GENERICV91A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.1a"{{.*}} "-target-feature" "+sve" "-target-feature" "+sve2"
