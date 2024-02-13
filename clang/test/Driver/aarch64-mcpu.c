// RUN: %clang --target=aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang --target=aarch64 -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang --target=aarch64 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang --target=aarch64 -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang --target=aarch64_be -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang --target=aarch64_be -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// GENERIC: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang --target=arm64 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang --target=arm64 -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang --target=arm64 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang --target=arm64 -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// ARM64-GENERIC: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64_be -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// RUN: %clang --target=aarch64 -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// RUN: %clang --target=aarch64_be -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// GENERIC-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// We cannot check much for -mcpu=native, but it should be replaced by either generic or a valid
// Arm cpu string, depending on the host.
// RUN: %clang --target=arm64 -mcpu=native -### -c %s 2> %t.err || true
// RUN: FileCheck --input-file=%t.err -check-prefix=ARM64-NATIVE %s
// ARM64-NATIVE-NOT: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "native"

// RUN: %clang --target=arm64-apple-ios -arch arm64 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-IOS %s
// RUN: %clang --target=arm64-apple-ios -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-IOS %s
// ARM64-IOS: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "apple-a7"
// ARM64-IOS-SAME: "-target-feature" "+aes"

// RUN: %clang --target=arm64-apple-ios -arch arm64e -### -c %s 2>&1 | FileCheck -check-prefix=ARM64E-IOS %s
// RUN: %clang --target=arm64e-apple-ios -### -c %s 2>&1 | FileCheck -check-prefix=ARM64E-IOS %s
// ARM64E-IOS: "-cc1"{{.*}} "-triple" "arm64e{{.*}}" "-target-cpu" "apple-a12"

// RUN: %clang --target=arm64-apple-watchos -arch arm64_32 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64_32-WATCHOS %s
// RUN: %clang --target=arm64_32-apple-watchos -### -c %s 2>&1 | FileCheck -check-prefix=ARM64_32-WATCHOS %s
// ARM64_32-WATCHOS: "-cc1"{{.*}} "-triple" "arm64_32{{.*}}" "-target-cpu" "apple-s4"

// RUN: %clang --target=aarch64 -mcpu=cortex-a77  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A77 %s
// CORTEX-A77: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a77"
// RUN: %clang --target=aarch64 -mcpu=cortex-x1  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXX1 %s
// CORTEXX1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x1"
// RUN: %clang --target=aarch64 -mcpu=cortex-x1c  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXX1C %s
// CORTEXX1C: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x1c"
// RUN: %clang --target=aarch64 -mcpu=cortex-x3 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXX3 %s
// CORTEXX3: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x3"
// RUN: %clang --target=aarch64 -mcpu=cortex-x4 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-X4 %s
// CORTEX-X4: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x4"
// RUN: %clang --target=aarch64 -mcpu=cortex-a78  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXA78 %s
// CORTEXA78: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a78"
// RUN: %clang --target=aarch64 -mcpu=cortex-a78c  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A78C %s
// CORTEX-A78C: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a78c"
// RUN: %clang --target=aarch64 -mcpu=cortex-a715  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A715 %s
// CORTEX-A715: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a715"
// RUN: %clang --target=aarch64 -mcpu=cortex-a720  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A720 %s
// CORTEX-A720: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a720"
// RUN: %clang --target=aarch64 -mcpu=neoverse-e1  -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-E1 %s
// NEOVERSE-E1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-e1"
// RUN: %clang --target=aarch64 -mcpu=neoverse-v1  -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-V1 %s
// NEOVERSE-V1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-v1"
// RUN: %clang --target=aarch64 -mcpu=neoverse-v2  -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-V2 %s
// NEOVERSE-V2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-v2"
// RUN: %clang --target=aarch64 -mcpu=neoverse-n1 -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-N1 %s
// NEOVERSE-N1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-n1"
// RUN: %clang --target=aarch64 -mcpu=neoverse-n2 -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-N2 %s
// NEOVERSE-N2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-n2"
// RUN: %clang --target=aarch64 -mcpu=neoverse-512tvb -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-512TVB %s
// NEOVERSE-512TVB: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-512tvb"
// RUN: %clang --target=aarch64 -mcpu=cortex-a520 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A520 %s
// CORTEX-A520: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a520"

// RUN: %clang --target=aarch64 -mcpu=cortex-r82  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXR82 %s
// CORTEXR82: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-r82"

// RUN: %clang --target=aarch64 -mcpu=cobalt-100 -### -c %s 2>&1 | FileCheck -check-prefix=COBALT-100 %s
// COBALT-100: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-n2"

// RUN: %clang --target=aarch64 -mcpu=grace -### -c %s 2>&1 | FileCheck -check-prefix=GRACE %s
// GRACE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-v2"

// ================== Check whether -mcpu and -mtune accept mixed-case values.
// RUN: %clang --target=aarch64 -mcpu=Cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA53 %s
// RUN: %clang --target=aarch64 -mtune=Cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA53-TUNE %s
// CASE-INSENSITIVE-CA53: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a53"
// CASE-INSENSITIVE-CA53-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=cortex-A53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA53 %s
// RUN: %clang --target=arm64 -mtune=cortex-A53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA53-TUNE %s
// CASE-INSENSITIVE-ARM64-CA53: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a53"
// CASE-INSENSITIVE-ARM64-CA53-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=aarch64 -mcpu=CORTEX-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA57 %s
// RUN: %clang --target=aarch64 -mtune=CORTEX-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA57-TUNE %s
// CASE-INSENSITIVE-CA57: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a57"
// CASE-INSENSITIVE-CA57-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang --target=arm64 -mcpu=Cortex-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA57 %s
// RUN: %clang --target=arm64 -mtune=Cortex-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA57-TUNE %s
// CASE-INSENSITIVE-ARM64-CA57: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a57"
// CASE-INSENSITIVE-ARM64-CA57-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
