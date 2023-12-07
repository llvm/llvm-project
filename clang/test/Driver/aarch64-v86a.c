// RUN: %clang --target=aarch64 -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang --target=aarch64 -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// GENERICV86A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.6a"

// RUN: %clang --target=aarch64_be -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// GENERICV86A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.6a"
