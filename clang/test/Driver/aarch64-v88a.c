// RUN: %clang --target=aarch64 -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang --target=aarch64 -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// GENERICV88A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.8a"

// RUN: %clang --target=aarch64_be -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// GENERICV88A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.8a"
