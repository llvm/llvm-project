// RUN: %clang --target=aarch64_be -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-BE %s
// GENERICV8A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"{{.*}} "-target-feature" "+v8a"{{.*}} "-target-feature" "+neon"

// RUN: %clang --target=aarch64 -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A %s
// RUN: %clang --target=aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A %s
// GENERICV8A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"{{.*}} "-target-feature" "+v8a"{{.*}} "-target-feature" "+neon"{{.*}}

// RUN: %clang --target=arm64 -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV8A %s
// RUN: %clang --target=arm64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV8A %s
// RUN: %clang --target=arm64 -mlittle-endian -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV8A %s
// RUN: %clang --target=arm64 -mlittle-endian -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV8A %s
// ARM64-GENERICV8A: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"{{.*}} "-target-feature" "+v8a"{{.*}} "-target-feature" "+neon"
