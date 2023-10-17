// RUN: %clang --target=aarch64 -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang --target=aarch64 -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// GENERICV82A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.2a"

// RUN: %clang --target=aarch64_be -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang --target=aarch64_be -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// GENERICV82A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.2a"
