// RUN: %clang --target=aarch64 -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang --target=aarch64 -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// GENERICV84A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.4a"

// RUN: %clang --target=aarch64_be -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// GENERICV84A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.4a"
