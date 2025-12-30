// RUN: %clang --target=aarch64 -march=armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A %s
// RUN: %clang --target=aarch64 -march=armv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A %s
// GENERICV89A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v8.9a"{{.*}} "-target-feature" "+neon"
// RUN: %clang --target=aarch64_be -march=armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV89A-BE %s
// GENERICV89A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v8.9a"{{.*}} "-target-feature" "+neon"
