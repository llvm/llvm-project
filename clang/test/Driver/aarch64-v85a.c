// RUN: %clang --target=aarch64 -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang --target=aarch64 -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// GENERICV85A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.5a"

// RUN: %clang --target=aarch64_be -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// GENERICV85A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.5a"
