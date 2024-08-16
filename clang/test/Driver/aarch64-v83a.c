// RUN: %clang --target=aarch64 -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64 -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// GENERICV83A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+v8.3a"{{.*}} "-target-feature" "+neon"

// RUN: %clang --target=aarch64_be -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// GENERICV83A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v8.3a"{{.*}} "-target-feature" "+neon"
