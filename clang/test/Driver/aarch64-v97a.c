// ===== Base v9.7a architecture =====

// RUN: %clang -target aarch64 -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// GENERICV97A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}}

// RUN: %clang -target aarch64_be -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// GENERICV97A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}}

// ===== Features supported on aarch64 =====

// RUN: %clang -target aarch64 -march=armv9.7a+cmh -### -c %s 2>&1 | FileCheck -check-prefix=V97A-CMH %s
// RUN: %clang -target aarch64 -march=armv9.7-a+cmh -### -c %s 2>&1 | FileCheck -check-prefix=V97A-CMH %s
// V97A-CMH: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+cmh"

// RUN: %clang -target aarch64 -march=armv9.7a+lscp -### -c %s 2>&1 | FileCheck -check-prefix=V97A-LSCP %s
// RUN: %clang -target aarch64 -march=armv9.7-a+lscp -### -c %s 2>&1 | FileCheck -check-prefix=V97A-LSCP %s
// V97A-LSCP: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+lscp"

// RUN: %clang -target aarch64 -march=armv9.7a+tlbid -### -c %s 2>&1 | FileCheck -check-prefix=V97A-TLBID %s
// RUN: %clang -target aarch64 -march=armv9.7-a+tlbid -### -c %s 2>&1 | FileCheck -check-prefix=V97A-TLBID %s
// V97A-TLBID: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+tlbid"
