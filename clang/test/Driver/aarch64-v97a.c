// ===== Base v9.7a architecture =====

// RUN: %clang -target aarch64 -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A %s
// GENERICV97A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+fprcvt"{{.*}} "-target-feature" "+sve2p3"{{.*}}

// RUN: %clang -target aarch64_be -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV97A-BE %s
// GENERICV97A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+fprcvt"{{.*}} "-target-feature" "+sve2p3"{{.*}}

// ===== Features supported on aarch64 =====

// RUN: %clang -target aarch64 -march=armv9.7a+sme2p3 -### -c %s 2>&1 | FileCheck -check-prefix=V97A-SME2p3 %s
// RUN: %clang -target aarch64 -march=armv9.7-a+sme2p3 -### -c %s 2>&1 | FileCheck -check-prefix=V97A-SME2p3 %s
// V97A-SME2p3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+sme2p3"

// RUN: %clang -target aarch64 -march=armv9.7a+sve2p3 -### -c %s 2>&1 | FileCheck -check-prefix=V97A-SVE2p3 %s
// RUN: %clang -target aarch64 -march=armv9.7-a+sve2p3 -### -c %s 2>&1 | FileCheck -check-prefix=V97A-SVE2p3 %s
// V97A-SVE2p3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+sve2p3"

// RUN: %clang -target aarch64 -march=armv9.7a+sve-b16mm -### -c %s 2>&1 | FileCheck -check-prefix=V97A-SVE-B16MM %s
// RUN: %clang -target aarch64 -march=armv9.7-a+sve-b16mm -### -c %s 2>&1 | FileCheck -check-prefix=V97A-SVE-B16MM %s
// V97A-SVE-B16MM: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+sve-b16mm"

// RUN: %clang -target aarch64 -march=armv9.7a+f16mm -### -c %s 2>&1 | FileCheck -check-prefix=V97A-F16MM %s
// RUN: %clang -target aarch64 -march=armv9.7-a+f16mm -### -c %s 2>&1 | FileCheck -check-prefix=V97A-F16MM %s
// V97A-F16MM: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+f16mm"

// RUN: %clang -target aarch64 -march=armv9.7a+cmh -### -c %s 2>&1 | FileCheck -check-prefix=V97A-CMH %s
// RUN: %clang -target aarch64 -march=armv9.7-a+cmh -### -c %s 2>&1 | FileCheck -check-prefix=V97A-CMH %s
// V97A-CMH: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+cmh"

// RUN: %clang -target aarch64 -march=armv9.7a+lscp -### -c %s 2>&1 | FileCheck -check-prefix=V97A-LSCP %s
// RUN: %clang -target aarch64 -march=armv9.7-a+lscp -### -c %s 2>&1 | FileCheck -check-prefix=V97A-LSCP %s
// V97A-LSCP: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+lscp"

// RUN: %clang -target aarch64 -march=armv9.7a+tlbid -### -c %s 2>&1 | FileCheck -check-prefix=V97A-TLBID %s
// RUN: %clang -target aarch64 -march=armv9.7-a+tlbid -### -c %s 2>&1 | FileCheck -check-prefix=V97A-TLBID %s
// V97A-TLBID: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+tlbid"

// RUN: %clang -target aarch64 -march=armv9.7a+mpamv2 -### -c %s 2>&1 | FileCheck -check-prefix=V97A-MPAMv2 %s
// RUN: %clang -target aarch64 -march=armv9.7-a+mpamv2 -### -c %s 2>&1 | FileCheck -check-prefix=V97A-MPAMv2 %s
// V97A-MPAMv2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+mpamv2"

// RUN: %clang -target aarch64 -march=armv9.7a+mtetc -### -c %s 2>&1 | FileCheck -check-prefix=V97A-MTETC %s
// RUN: %clang -target aarch64 -march=armv9.7-a+mtetc -### -c %s 2>&1 | FileCheck -check-prefix=V97A-MTETC %s
// V97A-MTETC: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+mtetc"

// RUN: %clang -target aarch64 -march=armv9.7a+gcie -### -c %s 2>&1 | FileCheck -check-prefix=V97A-GCIE %s
// RUN: %clang -target aarch64 -march=armv9.7-a+gcie -### -c %s 2>&1 | FileCheck -check-prefix=V97A-GCIE %s
// V97A-GCIE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.7a"{{.*}} "-target-feature" "+gcie"
