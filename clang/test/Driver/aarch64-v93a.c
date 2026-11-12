// RUN: %clang --target=aarch64 -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang --target=aarch64 -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// GENERICV93A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.3a"

// RUN: %clang --target=aarch64_be -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang --target=aarch64_be -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// GENERICV93A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.3a"
