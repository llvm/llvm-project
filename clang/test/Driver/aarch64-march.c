// RUN: %clang --target=aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-V8A %s
// RUN: %clang --target=aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-V8A %s
// GENERIC-V8A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8a"

// RUN: %clang --target=aarch64 -march=armv8-r -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-V8R %s
// GENERIC-V8R: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8r"

// RUN: %clang --target=aarch64 -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64 -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64 -mlittle-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang --target=aarch64_be -mlittle-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// GENERICV9A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9a" "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang --target=aarch64_be -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64_be -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// GENERICV9A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9a" "-target-feature" "+sve" "-target-feature" "+sve2"

// ================== Check whether -march accepts mixed-case values.
// RUN: %clang --target=aarch64_be -march=ARMV8.1A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64_be -march=ARMV8.1-A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=Armv8.1A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -march=Armv8.1-A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=ARMv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -march=ARMV8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// GENERICV81A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.1a"
