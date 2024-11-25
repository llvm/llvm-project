// ===== Base v9.6a architecture =====

// RUN: %clang -target aarch64 -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64 -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A %s
// GENERICV96A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+cmpbr"{{.*}} "-target-feature" "+fprcvt"{{.*}} "-target-feature" "+sve2p2"

// RUN: %clang -target aarch64_be -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV96A-BE %s
// GENERICV96A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+cmpbr"{{.*}} "-target-feature" "+fprcvt"{{.*}} "-target-feature" "+sve2p2"

// ===== Features supported on aarch64 =====

// RUN: %clang -target aarch64 -march=armv9.6a+f8f16mm -### -c %s 2>&1 | FileCheck -check-prefix=V96A-F8F16MM %s
// RUN: %clang -target aarch64 -march=armv9.6-a+f8f16mm -### -c %s 2>&1 | FileCheck -check-prefix=V96A-F8F16MM %s
// V96A-F8F16MM: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+f8f16mm"

// RUN: %clang -target aarch64 -march=armv9.6a+f8f32mm -### -c %s 2>&1 | FileCheck -check-prefix=V96A-F8F32MM %s
// RUN: %clang -target aarch64 -march=armv9.6-a+f8f32mm -### -c %s 2>&1 | FileCheck -check-prefix=V96A-F8F32MM %s
// V96A-F8F32MM: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+f8f32mm"

// RUN: %clang -target aarch64 -march=armv9.6a+lsfe -### -c %s 2>&1 | FileCheck -check-prefix=V96A-LSFE %s
// RUN: %clang -target aarch64 -march=armv9.6-a+lsfe -### -c %s 2>&1 | FileCheck -check-prefix=V96A-LSFE %s
// V96A-LSFE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+lsfe"

// RUN: %clang -target aarch64 -march=armv9.6a+sme2p2 -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SME2p2 %s
// RUN: %clang -target aarch64 -march=armv9.6-a+sme2p2 -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SME2p2 %s
// V96A-SME2p2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+sme2p2"

// RUN: %clang -target aarch64 -march=armv9.6a+ssve-aes -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SSVE-AES %s
// RUN: %clang -target aarch64 -march=armv9.6-a+ssve-aes -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SSVE-AES %s
// V96A-SSVE-AES: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+ssve-aes"

// RUN: %clang -target aarch64 -march=armv9.6a+sve2p2 -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE2p2 %s
// RUN: %clang -target aarch64 -march=armv9.6-a+sve2p2 -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE2p2 %s
// V96A-SVE2p2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+sve2p2"

// RUN: %clang -target aarch64 -march=armv9.6a+sve-aes2 -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE-AES2 %s
// RUN: %clang -target aarch64 -march=armv9.6-a+sve-aes2 -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE-AES2 %s
// V96A-SVE-AES2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+sve-aes2"

// RUN: %clang -target aarch64 -march=armv9.6a+sve-bfscale -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE-BFSCALE %s
// RUN: %clang -target aarch64 -march=armv9.6-a+sve-bfscale -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE-BFSCALE %s
// V96A-SVE-BFSCALE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+sve-bfscale"

// RUN: %clang -target aarch64 -march=armv9.6a+sve-f16f32mm -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE-F16F32MM %s
// RUN: %clang -target aarch64 -march=armv9.6-a+sve-f16f32mm -### -c %s 2>&1 | FileCheck -check-prefix=V96A-SVE-F16F32MM %s
// V96A-SVE-F16F32MM: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+sve-f16f32mm"
//
// RUN: %clang -target aarch64 -march=armv9.6a+lsui -### -c %s 2>&1 | FileCheck -check-prefix=V96A-LSUI %s
// RUN: %clang -target aarch64 -march=armv9.6-a+lsui -### -c %s 2>&1 | FileCheck -check-prefix=V96A-LSUI %s
// V96A-LSUI: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+lsui"
//
// RUN: %clang -target aarch64 -march=armv9.6a+occmo -### -c %s 2>&1 | FileCheck -check-prefix=V96A-OCCMO %s
// RUN: %clang -target aarch64 -march=armv9.6-a+occmo -### -c %s 2>&1 | FileCheck -check-prefix=V96A-OCCMO %s
// V96A-OCCMO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+occmo"
//
// RUN: %clang -target aarch64 -march=armv9.6a+pcdphint -### -c %s 2>&1 | FileCheck -check-prefix=V96A-PCDPHINT %s
// RUN: %clang -target aarch64 -march=armv9.6-a+pcdphint -### -c %s 2>&1 | FileCheck -check-prefix=V96A-PCDPHINT %s
// V96A-PCDPHINT: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+pcdphint"

// RUN: %clang -target aarch64 -march=armv9.6a+pops -### -c %s 2>&1 | FileCheck -check-prefix=V96A-POPS %s
// RUN: %clang -target aarch64 -march=armv9.6-a+pops -### -c %s 2>&1 | FileCheck -check-prefix=V96A-POPS %s
// V96A-POPS: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+v9.6a"{{.*}} "-target-feature" "+pops"
//
