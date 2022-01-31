// Check target CPUs are correctly passed.

// RUN: %clang -target aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64_be -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// GENERIC: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang -target arm64 -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang -target arm64 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s

// ARM64-GENERIC: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// We cannot check much for -mcpu=native, but it should be replaced by either generic or a valid
// Arm cpu string, depending on the host.
// RUN: %clang -target arm64 -mcpu=native -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-NATIVE %s
// ARM64-NATIVE-NOT: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "native"

// RUN: %clang -target arm64-apple-ios -arch arm64 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-IOS %s
// RUN: %clang -target arm64-apple-ios -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-IOS %s
// ARM64-IOS: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "apple-a7"
// ARM64-IOS-SAME: "-target-feature" "+aes"

// RUN: %clang -target arm64-apple-ios -arch arm64e -### -c %s 2>&1 | FileCheck -check-prefix=ARM64E-IOS %s
// RUN: %clang -target arm64e-apple-ios -### -c %s 2>&1 | FileCheck -check-prefix=ARM64E-IOS %s
// ARM64E-IOS: "-cc1"{{.*}} "-triple" "arm64e{{.*}}" "-target-cpu" "apple-a12"

// RUN: %clang -target arm64-apple-watchos -arch arm64_32 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64_32-WATCHOS %s
// RUN: %clang -target arm64_32-apple-watchos -### -c %s 2>&1 | FileCheck -check-prefix=ARM64_32-WATCHOS %s
// ARM64_32-WATCHOS: "-cc1"{{.*}} "-triple" "arm64_32{{.*}}" "-target-cpu" "apple-s4"

// RUN: %clang -target aarch64 -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35 %s
// RUN: %clang -target aarch64 -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-TUNE %s
// CA35: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a35"
// CA35-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35 %s
// RUN: %clang -target arm64 -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA35-TUNE %s
// ARM64-CA35: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a35"
// ARM64-CA35-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34 %s
// RUN: %clang -target aarch64 -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-TUNE %s
// CA34: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a34"
// CA34-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34 %s
// RUN: %clang -target arm64 -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA34-TUNE %s
// ARM64-CA34: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a34"
// ARM64-CA34-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-TUNE %s
// CA53: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a53"
// CA53-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53 %s
// RUN: %clang -target arm64 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53-TUNE %s
// ARM64-CA53: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a53"
// ARM64-CA53-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55 %s
// RUN: %clang -target aarch64 -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-TUNE %s
// CA55: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a55"
// CA55-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55 %s
// RUN: %clang -target arm64 -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA55-TUNE %s
// ARM64-CA55: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a55"
// ARM64-CA55-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64 -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-TUNE %s
// CA57: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a57"
// CA57-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57 %s
// RUN: %clang -target arm64 -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57-TUNE %s
// ARM64-CA57: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a57"
// ARM64-CA57-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64 -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-TUNE %s
// CA72: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a72"
// CA72-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// RUN: %clang -target arm64 -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72-TUNE %s
// ARM64-CA72: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a72"
// ARM64-CA72-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73 %s
// RUN: %clang -target aarch64 -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-TUNE %s
// CORTEX-A73: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a73"
// CORTEX-A73-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A73 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A73 %s
// RUN: %clang -target arm64 -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A73-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A73-TUNE %s
// ARM64-CORTEX-A73: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a73"
// ARM64-CORTEX-A73-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75 %s
// RUN: %clang -target aarch64 -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A75-TUNE %s
// CORTEX-A75: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a75"
// CORTEX-A75-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75 %s
// RUN: %clang -target arm64 -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a75 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A75-TUNE %s
// ARM64-CORTEX-A75: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a75"
// ARM64-CORTEX-A75-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76 %s
// RUN: %clang -target aarch64 -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A76-TUNE %s
// CORTEX-A76: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "cortex-a76"
// CORTEX-A76-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76 %s
// RUN: %clang -target arm64 -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a76 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CORTEX-A76-TUNE %s
// ARM64-CORTEX-A76: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a76"
// ARM64-CORTEX-A76-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a77  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A77 %s
// CORTEX-A77: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a77"
// RUN: %clang -target aarch64 -mcpu=cortex-x1  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXX1 %s
// CORTEXX1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x1"
// RUN: %clang -target aarch64 -mcpu=cortex-x1c  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXX1C %s
// CORTEXX1C: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x1c"
// RUN: %clang -target aarch64 -mcpu=cortex-a78  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXA78 %s
// CORTEXA78: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a78"
// RUN: %clang -target aarch64 -mcpu=cortex-a78c  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A78C %s
// CORTEX-A78C: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a78c"
// RUN: %clang -target aarch64 -mcpu=neoverse-e1  -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-E1 %s
// NEOVERSE-E1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-e1"
// RUN: %clang -target aarch64 -mcpu=neoverse-v1  -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-V1 %s
// NEOVERSE-V1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-v1"
// RUN: %clang -target aarch64 -mcpu=neoverse-n1 -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-N1 %s
// NEOVERSE-N1: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-n1"
// RUN: %clang -target aarch64 -mcpu=neoverse-n2 -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-N2 %s
// NEOVERSE-N2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-n2"

// RUN: %clang -target aarch64 -mcpu=neoverse-512tvb -### -c %s 2>&1 | FileCheck -check-prefix=NEOVERSE-512TVB %s
// NEOVERSE-512TVB: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "neoverse-512tvb"

// RUN: %clang -target aarch64 -mcpu=cortex-r82  -### -c %s 2>&1 | FileCheck -check-prefix=CORTEXR82 %s
// CORTEXR82: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-r82"

// RUN: %clang -target aarch64_be -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3 %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3 %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3 %s
// RUN: %clang -target aarch64_be -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-TUNE %s
// M3: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m3"
// M3-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4 %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4 %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4 %s
// RUN: %clang -target aarch64_be -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-TUNE %s
// M4: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m4" "-target-feature" "+v8.2a"
// M4-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"
// M4-TUNE-NOT: "+v8.2a"

// RUN: %clang -target aarch64_be -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5 %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5 %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5 %s
// RUN: %clang -target aarch64_be -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-TUNE %s
// M5: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m5" "-target-feature" "+v8.2a"
// M5-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"
// M5-TUNE-NOT: "+v8.2a"

// RUN: %clang -target arm64 -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3 %s
// RUN: %clang -target arm64 -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M3-TUNE %s
// ARM64-M3: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "exynos-m3"
// ARM64-M3-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4 %s
// RUN: %clang -target arm64 -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M4-TUNE %s
// ARM64-M4: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "exynos-m4" "-target-feature" "+v8.2a"
// ARM64-M4-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
// ARM64-M4-TUNE-NOT: "+v8.2a"

// RUN: %clang -target arm64 -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5 %s
// RUN: %clang -target arm64 -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-M5-TUNE %s
// ARM64-M5: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "exynos-m5" "-target-feature" "+v8.2a"
// ARM64-M5-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
// ARM64-M5-TUNE-NOT: "+v8.2a"

// RUN: %clang -target aarch64 -mcpu=falkor -### -c %s 2>&1 | FileCheck -check-prefix=FALKOR %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=falkor -### -c %s 2>&1 | FileCheck -check-prefix=FALKOR %s
// RUN: %clang -target aarch64 -mtune=falkor -### -c %s 2>&1 | FileCheck -check-prefix=FALKOR-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=falkor -### -c %s 2>&1 | FileCheck -check-prefix=FALKOR-TUNE %s
// FALKOR: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "falkor"
// FALKOR-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=falkor -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-FALKOR %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=falkor -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-FALKOR %s
// RUN: %clang -target arm64 -mtune=falkor -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-FALKOR-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=falkor -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-FALKOR-TUNE %s
// ARM64-FALKOR: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "falkor"
// ARM64-FALKOR-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=kryo -### -c %s 2>&1 | FileCheck -check-prefix=KRYO %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=kryo -### -c %s 2>&1 | FileCheck -check-prefix=KRYO %s
// RUN: %clang -target aarch64 -mtune=kryo -### -c %s 2>&1 | FileCheck -check-prefix=KRYO-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=kryo -### -c %s 2>&1 | FileCheck -check-prefix=KRYO-TUNE %s
// KRYO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "kryo"
// KRYO-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=kryo -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-KRYO %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=kryo -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-KRYO %s
// RUN: %clang -target arm64 -mtune=kryo -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-KRYO-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=kryo -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-KRYO-TUNE %s
// ARM64-KRYO: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "kryo"
// ARM64-KRYO-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99 %s
// RUN: %clang -target aarch64 -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-TUNE %s
// THUNDERX2T99: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "thunderx2t99" "-target-feature" "+v8.1a"
// THUNDERX2T99-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"
// THUNDERX2T99-TUNE-NOT: +v8.1a

// RUN: %clang -target aarch64 -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110 %s
// RUN: %clang -target aarch64 -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-TUNE %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-TUNE %s
// THUNDERX3T110: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "thunderx3t110" "-target-feature" "+v8.3a"
// THUNDERX3T110-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX2T99 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX2T99 %s
// RUN: %clang -target arm64 -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX2T99-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX2T99-TUNE %s
// ARM64-THUNDERX2T99: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "thunderx2t99" "-target-feature" "+v8.1a"
// ARM64-THUNDERX2T99-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
// ARM64-THUNDERX2T99-TUNE-NOT: +v8.1a

// RUN: %clang -target arm64 -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110 %s
// RUN: %clang -target arm64 -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-THUNDERX3T110-TUNE %s
// ARM64-THUNDERX3T110: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "thunderx3t110" "-target-feature" "+v8.3a"
// ARM64-THUNDERX3T110-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110 %s
// RUN: %clang -target aarch64 -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=TSV110-TUNE %s
// TSV110: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "tsv110"
// TSV110-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110 %s
// RUN: %clang -target arm64 -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=tsv110 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-TSV110-TUNE %s
// ARM64-TSV110: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "tsv110"
// ARM64-TSV110-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX %s
// RUN: %clang -target aarch64 -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=A64FX-TUNE %s
// A64FX: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "a64fx"
// A64FX-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX %s
// RUN: %clang -target arm64 -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=a64fx -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-A64FX-TUNE %s
// ARM64-A64FX: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "a64fx"
// ARM64-A64FX-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL %s
// RUN: %clang -target aarch64 -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL-TUNE %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=CARMEL-TUNE %s
// CARMEL: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "carmel"
// CARMEL-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL %s
// RUN: %clang -target arm64 -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL-TUNE %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=carmel -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CARMEL-TUNE %s
// ARM64-CARMEL: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "carmel"
// ARM64-CARMEL-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// RUN: %clang -target aarch64 -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// GENERIC-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a35 -### -c %s 2>&1 | FileCheck -check-prefix=CA35-BE-TUNE %s
// CA35-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a35"
// CA35-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a34 -### -c %s 2>&1 | FileCheck -check-prefix=CA34-BE-TUNE %s
// CA34-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a34"
// CA34-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE-TUNE %s
// CA53-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a53"
// CA53-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a55 -### -c %s 2>&1 | FileCheck -check-prefix=CA55-BE-TUNE %s
// CA55-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a55"
// CA55-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a510 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A510 %s
// CORTEX-A510: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a510"
// CORTEX-A510-NOT: "-target-feature" "{{[+-]}}sm4"
// CORTEX-A510-NOT: "-target-feature" "{{[+-]}}sha3"
// CORTEX-A510-NOT: "-target-feature" "{{[+-]}}aes"
// CORTEX-A510-SAME: {{$}}
// RUN: %clang -target aarch64 -mcpu=cortex-a510+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A510-CRYPTO %s
// CORTEX-A510-CRYPTO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+sm4" "-target-feature" "+sha3" "-target-feature" "+sha2" "-target-feature" "+aes"

// RUN: %clang -target aarch64 -mcpu=cortex-x2 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-X2 %s
// CORTEX-X2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-x2"
// CORTEX-X2-NOT: "-target-feature" "{{[+-]}}sm4"
// CORTEX-X2-NOT: "-target-feature" "{{[+-]}}sha3"
// CORTEX-X2-NOT: "-target-feature" "{{[+-]}}aes"
// CORTEX-X2-SAME: {{$}}
// RUN: %clang -target aarch64 -mcpu=cortex-x2+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-X2-CRYPTO %s
// CORTEX-X2-CRYPTO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+sm4" "-target-feature" "+sha3" "-target-feature" "+sha2" "-target-feature" "+aes"

// RUN: %clang -target aarch64 -mcpu=cortex-a710 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A710 %s
// CORTEX-A710: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a710"
// CORTEX-A710-NOT: "-target-feature" "{{[+-]}}sm4"
// CORTEX-A710-NOT: "-target-feature" "{{[+-]}}sha3"
// CORTEX-A710-NOT: "-target-feature" "{{[+-]}}aes"
// CORTEX-A710-SAME: {{$}}
// RUN: %clang -target aarch64 -mcpu=cortex-a710+crypto -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A710-CRYPTO %s
// CORTEX-A710-CRYPTO: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-feature" "+sm4" "-target-feature" "+sha3" "-target-feature" "+sha2" "-target-feature" "+aes"

// RUN: %clang -target aarch64_be -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE-TUNE %s
// CA57-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a57"
// CA57-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE-TUNE %s
// CA72-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a72"
// CA72-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a73 -### -c %s 2>&1 | FileCheck -check-prefix=CORTEX-A73-BE-TUNE %s
// CORTEX-A73-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a73"
// CORTEX-A73-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE %s
// RUN: %clang -target aarch64_be -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=exynos-m3 -### -c %s 2>&1 | FileCheck -check-prefix=M3-BE-TUNE %s
// M3-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m3"
// M3-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE %s
// RUN: %clang -target aarch64_be -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=exynos-m4 -### -c %s 2>&1 | FileCheck -check-prefix=M4-BE-TUNE %s
// M4-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m4" "-target-feature" "+v8.2a"
// M4-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
// M4-BE-TUNE-NOT: "+v8.2a"

// RUN: %clang -target aarch64_be -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE %s
// RUN: %clang -target aarch64_be -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=exynos-m5 -### -c %s 2>&1 | FileCheck -check-prefix=M5-BE-TUNE %s
// M5-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "exynos-m5" "-target-feature" "+v8.2a"
// M5-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"
// M5-BE-TUNE-NOT: "+v8.2a"

// RUN: %clang -target aarch64_be -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-BE %s
// RUN: %clang -target aarch64_be -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=thunderx2t99 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX2T99-BE-TUNE %s
// THUNDERX2T99-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "thunderx2t99"
// THUNDERX2T99-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE %s
// RUN: %clang -target aarch64_be -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE-TUNE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE-TUNE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=thunderx3t110 -### -c %s 2>&1 | FileCheck -check-prefix=THUNDERX3T110-BE-TUNE %s
// THUNDERX3T110-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "thunderx3t110"
// THUNDERX3T110-BE-TUNE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=cortex-a57 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-A57 %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=cortex-a57  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-A57 %s
// RUN: %clang -target aarch64 -mcpu=cortex-a72 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-A72 %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=cortex-a72  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-A72 %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=cortex-a73     -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-A73 %s
// RUN: %clang -target aarch64 -mcpu=thunderx2t99 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-THUNDERX2T99 %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=thunderx2t99  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-THUNDERX2T99 %s
// RUN: %clang -target aarch64 -mcpu=thunderx3t110 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-THUNDERX3T110 %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=thunderx3t110  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE-THUNDERX3T110 %s
// MCPU-MTUNE-A57: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a57"
// MCPU-MTUNE-A72: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a72"
// MCPU-MTUNE-A73: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a73"
// MCPU-MTUNE-THUNDERX2T99: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "thunderx2t99"
// MCPU-MTUNE-THUNDERX3T110: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "thunderx3t110"

// RUN: %clang -target aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-V8A %s
// RUN: %clang -target aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-V8A %s
// GENERIC-V8A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8a"

// RUN: %clang -target aarch64 -march=armv8-r -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-V8R %s
// GENERIC-V8R: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8r"

// RUN: %clang -target aarch64 -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64 -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// GENERICV81A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.1a"

// RUN: %clang -target arm64 -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// RUN: %clang -target arm64 -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// RUN: %clang -target arm64 -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// RUN: %clang -target arm64 -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERICV81A %s
// ARM64-GENERICV81A: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.1a"

// RUN: %clang -target aarch64_be -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s

// RUN: %clang -target aarch64 -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang -target aarch64 -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A %s
// GENERICV82A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.2a"

// RUN: %clang -target aarch64_be -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-BE %s
// GENERICV82A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.2a"

// RUN: %clang -target aarch64 -march=armv8.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML %s
// GENERICV82A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV82A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang -target aarch64 -march=armv8.2a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.2-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16 %s
// GENERICV82A-FP16: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.2a" "-target-feature" "+fullfp16"
// GENERICV82A-FP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV82A-FP16-SAME: {{$}}

// RUN: %clang -target aarch64 -march=armv8.2-a+profile -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-SPE %s
// GENERICV82A-SPE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.2a" "-target-feature" "+spe"

// RUN: %clang -target aarch64 -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NO-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NO-FP16FML %s
// GENERICV8A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV8A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang -target aarch64 -march=armv8a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16 %s
// RUN: %clang -target aarch64 -march=armv8-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16 %s
// GENERICV8A-FP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV8A-FP16: "-target-feature" "+fullfp16"
// GENERICV8A-FP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV8A-FP16-SAME: {{$}}

// RUN: %clang -target aarch64 -march=armv8a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-FP16FML %s
// GENERICV8A-FP16FML: "-target-feature" "+fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.2a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.2-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML %s
// GENERICV82A-FP16FML: "-target-feature" "+fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.2a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-NO-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.2-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-NO-FP16FML %s
// GENERICV82A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang -target aarch64 -march=armv8.2a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.2-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16FML-FP16 %s
// GENERICV82A-NO-FP16FML-FP16: "-target-feature" "-fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.2a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML-NO-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.2-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16FML-NO-FP16 %s
// GENERICV82A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang -target aarch64 -march=armv8.2a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.2-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-NO-FP16-FP16FML %s
// GENERICV82A-NO-FP16-FP16FML: "-target-feature" "+fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.2a+fp16+profile -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-SPE %s
// RUN: %clang -target aarch64 -march=armv8.2-a+fp16+profile -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV82A-FP16-SPE %s
// GENERICV82A-FP16-SPE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.2a" "-target-feature" "+fullfp16" "-target-feature" "+spe"
// GENERICV82A-FP16-SPE-NOT:  "-target-feature" "{{[+-]}}fp16fml"
// GENERICV82A-FP16-SPE-SAME: {{$}}

// RUN: %clang -target aarch64 -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang -target aarch64 -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A %s
// GENERICV83A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.3a"

// RUN: %clang -target aarch64_be -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-BE %s
// GENERICV83A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.3a"

// RUN: %clang -target aarch64 -march=armv8.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML %s
// GENERICV83A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV83A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang -target aarch64 -march=armv8.3a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.3-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16 %s
// GENERICV83A-FP16: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.3a" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.3a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.3-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML %s
// GENERICV83A-FP16FML: "-target-feature" "+fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.3a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16-NO-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.3-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16-NO-FP16FML %s
// GENERICV83A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang -target aarch64 -march=armv8.3a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.3-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16FML-FP16 %s
// GENERICV83A-NO-FP16FML-FP16: "-target-feature" "-fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.3a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML-NO-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.3-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-FP16FML-NO-FP16 %s
// GENERICV83A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang -target aarch64 -march=armv8.3a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.3-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV83A-NO-FP16-FP16FML %s
// GENERICV83A-NO-FP16-FP16FML: "-target-feature" "+fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang -target aarch64 -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A %s
// GENERICV84A: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.4a"

// RUN: %clang -target aarch64_be -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-BE %s
// GENERICV84A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.4a"

// RUN: %clang -target aarch64 -march=armv8.4a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.4-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML %s
// GENERICV84A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fp16fml"
// GENERICV84A-NO-FP16FML-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RUN: %clang -target aarch64 -march=armv8.4a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.4-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16 %s
// GENERICV84A-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang -target aarch64 -march=armv8.4a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.4-a+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML %s
// GENERICV84A-FP16FML: "-target-feature" "+fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.4a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16-NO-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.4-a+fp16+nofp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16-NO-FP16FML %s
// GENERICV84A-FP16-NO-FP16FML: "-target-feature" "+fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang -target aarch64 -march=armv8.4a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.4-a+nofp16fml+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16FML-FP16 %s
// GENERICV84A-NO-FP16FML-FP16: "-target-feature" "+fullfp16" "-target-feature" "+fp16fml"

// RUN: %clang -target aarch64 -march=armv8.4a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML-NO-FP16 %s
// RUN: %clang -target aarch64 -march=armv8.4-a+fp16fml+nofp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-FP16FML-NO-FP16 %s
// GENERICV84A-FP16FML-NO-FP16: "-target-feature" "-fullfp16" "-target-feature" "-fp16fml"

// RUN: %clang -target aarch64 -march=armv8.4a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16-FP16FML %s
// RUN: %clang -target aarch64 -march=armv8.4-a+nofp16+fp16fml -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV84A-NO-FP16-FP16FML %s
// GENERICV84A-NO-FP16-FP16FML: "-target-feature" "+fp16fml" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang -target aarch64 -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A %s
// GENERICV85A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.5a"

// RUN: %clang -target aarch64_be -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.5-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BE %s
// GENERICV85A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.5a"

// RUN: %clang -target aarch64 -march=armv8.5-a+fp16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-FP16 %s
// GENERICV85A-FP16: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.5a" "-target-feature" "+fullfp16"

// RUN: %clang -target aarch64 -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang -target aarch64 -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A %s
// GENERICV86A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.6a"

// RUN: %clang -target aarch64_be -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.6-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV86A-BE %s
// GENERICV86A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.6a"

// The SVE extension is an optional extension for Armv8-A.
// RUN: %clang -target aarch64 -march=armv8a+sve -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-SVE %s
// RUN: %clang -target aarch64 -march=armv8.6a+sve -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-SVE %s
// GENERICV8A-SVE: "-target-feature" "+sve"
// RUN: %clang -target aarch64 -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NOSVE %s
// RUN: %clang -target aarch64 -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NOSVE %s
// GENERICV8A-NOSVE-NOT: "-target-feature" "+sve"

// The BFloat16 extension is a mandatory component of the Armv8.6-A extensions, but is permitted as an
// optional feature for any implementation of Armv8.2-A to Armv8.5-A (inclusive)
// RUN: %clang -target aarch64 -march=armv8.5a+bf16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BF16 %s
// GENERICV85A-BF16: "-target-feature" "+bf16"
// RUN: %clang -target aarch64 -march=armv8.5a+bf16+nobf16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BF16-NO-BF16 %s
// GENERICV85A-BF16-NO-BF16: "-target-feature" "-bf16"
// RUN: %clang -target aarch64 -march=armv8.5a+bf16+sve -### -c %s 2>&1 | FileCheck -check-prefixes=GENERICV85A-BF16-SVE %s
// GENERICV85A-BF16-SVE: "-target-feature" "+bf16" "-target-feature" "+sve"

// The 8-bit integer matrix multiply extension is a mandatory component of the
// Armv8.6-A extensions, but is permitted as an optional feature for any
// implementation of Armv8.2-A to Armv8.5-A (inclusive)
// RUN: %clang -target aarch64 -march=armv8.5a -### -c %s 2>&1 | FileCheck -check-prefix=NO-I8MM %s
// RUN: %clang -target aarch64 -march=armv8.5a+i8mm -### -c %s 2>&1 | FileCheck -check-prefix=I8MM %s
// NO-I8MM-NOT: "-target-feature" "+i8mm"
// I8MM: "-target-feature" "+i8mm"

// The 32-bit floating point matrix multiply extension is enabled by default
// for armv8.6-a targets (or later) with SVE, and can optionally be enabled for
// any target from armv8.2a onwards (we don't enforce not using it with earlier
// targets).
// RUN: %clang -target aarch64 -march=armv8.6a       -### -c %s 2>&1 | FileCheck -check-prefix=NO-F32MM %s
// RUN: %clang -target aarch64 -march=armv8.6a+sve   -### -c %s 2>&1 | FileCheck -check-prefix=F32MM %s
// RUN: %clang -target aarch64 -march=armv8.5a+f32mm -### -c %s 2>&1 | FileCheck -check-prefix=F32MM %s
// NO-F32MM-NOT: "-target-feature" "+f32mm"
// F32MM: "-target-feature" "+f32mm"

// The 64-bit floating point matrix multiply extension is not currently enabled
// by default for any targets, because it requires an SVE vector length >= 256
// bits. When we add a CPU which has that, then it can be enabled by default,
// but for now it can only be used by adding the +f64mm feature.
// RUN: %clang -target aarch64 -march=armv8.6a       -### -c %s 2>&1 | FileCheck -check-prefix=NO-F64MM %s
// RUN: %clang -target aarch64 -march=armv8.6a+sve   -### -c %s 2>&1 | FileCheck -check-prefix=NO-F64MM %s
// RUN: %clang -target aarch64 -march=armv8.6a+f64mm -### -c %s 2>&1 | FileCheck -check-prefix=F64MM %s
// NO-F64MM-NOT: "-target-feature" "+f64mm"
// F64MM: "-target-feature" "+f64mm"

// RUN: %clang -target aarch64 -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64 -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A %s
// GENERICV87A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.7a"

// RUN: %clang -target aarch64_be -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.7a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.7-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV87A-BE %s
// GENERICV87A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.7a"

// Tha LD64B/ST64B accelerator extension is disabled by default.
// RUN: %clang -target aarch64 -march=armv8.7a       -### -c %s 2>&1 | FileCheck -check-prefix=NO-LS64 %s
// RUN: %clang -target aarch64 -march=armv8.7a+ls64 -### -c %s 2>&1 | FileCheck -check-prefix=LS64 %s
// NO-LS64-NOT: "-target-feature" "+ls64"
// LS64: "-target-feature" "+ls64"

// RUN: %clang -target aarch64 -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang -target aarch64 -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A %s
// GENERICV88A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.8a"

// RUN: %clang -target aarch64_be -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang -target aarch64_be -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv8.8-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV88A-BE %s
// GENERICV88A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.8a"
//
// RUN: %clang -target aarch64 -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang -target aarch64 -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A %s
// GENERICV9A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9a" "-target-feature" "+sve" "-target-feature" "+sve2"

// SVE2 is enabled by default on Armv9-A but it can be disabled
// RUN: %clang -target aarch64 -march=armv9a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang -target aarch64 -march=armv9-a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9-a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9-a+nosve2 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-NOSVE2 %s
// GENERICV9A-NOSVE2: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9a" "-target-feature" "+sve" "-target-feature" "-sve2"

// RUN: %clang -target aarch64_be -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang -target aarch64_be -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV9A-BE %s
// GENERICV9A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9a" "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang -target aarch64 -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang -target aarch64 -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A %s
// GENERICV91A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.1a" "-target-feature" "+i8mm" "-target-feature" "+bf16" "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang -target aarch64_be -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV91A-BE %s
// GENERICV91A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.1a" "-target-feature" "+i8mm" "-target-feature" "+bf16" "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang -target aarch64 -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang -target aarch64 -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A %s
// GENERICV92A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.2a" "-target-feature" "+i8mm" "-target-feature" "+bf16" "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang -target aarch64_be -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.2a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.2-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV92A-BE %s
// GENERICV92A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.2a" "-target-feature" "+i8mm" "-target-feature" "+bf16" "-target-feature" "+sve" "-target-feature" "+sve2"

// RUN: %clang -target aarch64 -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang -target aarch64 -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A %s
// GENERICV93A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.3a"

// RUN: %clang -target aarch64_be -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang -target aarch64_be -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.3a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=armv9.3-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV93A-BE %s
// GENERICV93A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v9.3a"

// fullfp16 is off by default for v8a, feature must not be mentioned
// RUN: %clang -target aarch64 -march=armv8a  -### -c %s 2>&1 | FileCheck -check-prefix=V82ANOFP16 -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -march=armv8-a -### -c %s 2>&1 | FileCheck -check-prefix=V82ANOFP16 -check-prefix=GENERIC %s
// V82ANOFP16-NOT: "-target-feature" "{{[+-]}}fp16fml"
// V82ANOFP16-NOT: "-target-feature" "{{[+-]}}fullfp16"

// RAS is on by default for v8.2a, but can be disabled by +noras
// RUN: %clang -target aarch64 -march=armv8.2a  -### -c %s 2>&1 | FileCheck -check-prefix=V82ARAS -check-prefix=GENERICV82A %s
// RUN: %clang -target aarch64 -march=armv8.2-a -### -c %s 2>&1 | FileCheck -check-prefix=V82ARAS -check-prefix=GENERICV82A %s
// V82ARAS-NOT: "-target-feature" "+ras"
// V82ARAS-NOT: "-target-feature" "-ras"
// RUN: %clang -target aarch64 -march=armv8.2a+noras  -### -c %s 2>&1 | FileCheck -check-prefix=V82ANORAS -check-prefix=GENERICV82A %s
// RUN: %clang -target aarch64 -march=armv8.2-a+noras -### -c %s 2>&1 | FileCheck -check-prefix=V82ANORAS -check-prefix=GENERICV82A %s
// V82ANORAS: "-target-feature" "-ras"

// RAS is off by default for v8a, but can be enabled by +ras (this is not architecturally valid)
// RUN: %clang -target aarch64 -march=armv8a+ras  -### -c %s 2>&1 | FileCheck -check-prefix=V8ARAS -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -march=armv8-a+ras -### -c %s 2>&1 | FileCheck -check-prefix=V8ARAS -check-prefix=GENERIC %s
// V8ARAS: "-target-feature" "+ras"

// ================== Check whether -march accepts mixed-case values.
// RUN: %clang -target aarch64_be -march=ARMV8.1A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64_be -march=ARMV8.1-A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=Armv8.1A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64 -mbig-endian -march=Armv8.1-A -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=ARMv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -march=ARMV8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A-BE %s
// GENERICV81A-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.1a"

// ================== Check whether -mcpu and -mtune accept mixed-case values.
// RUN: %clang -target aarch64 -mcpu=Cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA53 %s
// RUN: %clang -target aarch64 -mtune=Cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA53-TUNE %s
// CASE-INSENSITIVE-CA53: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a53"
// CASE-INSENSITIVE-CA53-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=cortex-A53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA53 %s
// RUN: %clang -target arm64 -mtune=cortex-A53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA53-TUNE %s
// CASE-INSENSITIVE-ARM64-CA53: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a53"
// CASE-INSENSITIVE-ARM64-CA53-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64 -mcpu=CORTEX-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA57 %s
// RUN: %clang -target aarch64 -mtune=CORTEX-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA57-TUNE %s
// CASE-INSENSITIVE-CA57: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a57"
// CASE-INSENSITIVE-CA57-TUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -mcpu=Cortex-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA57 %s
// RUN: %clang -target arm64 -mtune=Cortex-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA57-TUNE %s
// CASE-INSENSITIVE-ARM64-CA57: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a57"
// CASE-INSENSITIVE-ARM64-CA57-TUNE: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"
