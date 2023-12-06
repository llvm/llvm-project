// RUN: %clang -target aarch64 -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64 -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A %s
// GENERICV95A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.5a"
// RUN: %clang -target aarch64_be -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV95A-BE %s
// GENERICV95A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.5a"

