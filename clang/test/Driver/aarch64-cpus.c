// Check target CPUs are correctly passed.

// RUN: %clang -target aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64_be -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC %s
// GENERIC: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang -target arm64 -mcpu=generic -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang -target arm64 -mlittle-endian -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu-generic -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-GENERIC %s

// ARM64-GENERIC: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target arm64-apple-darwin -arch arm64 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-DARWIN %s
// ARM64-DARWIN: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cyclone"

// RUN: %clang -target aarch64 -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53 %s
// CA53: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a53"

// RUN: %clang -target arm64 -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53 %s
// RUN: %clang -target arm64 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53 %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA53 %s
// ARM64-CA53: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a53"

// RUN: %clang -target aarch64 -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64 -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57 %s
// CA57: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a57"

// RUN: %clang -target arm64 -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57 %s
// RUN: %clang -target arm64 -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57 %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA57 %s
// ARM64-CA57: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a57"

// RUN: %clang -target aarch64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64 -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64 -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64 -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// RUN: %clang -target aarch64_be -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72 %s
// CA72: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a72"

// RUN: %clang -target arm64 -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// RUN: %clang -target arm64 -mlittle-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// RUN: %clang -target arm64 -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// RUN: %clang -target arm64 -mlittle-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=ARM64-CA72 %s
// ARM64-CA72: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a72"

// RUN: %clang -target aarch64_be -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// RUN: %clang -target aarch64 -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -### -c %s 2>&1 | FileCheck -check-prefix=GENERIC-BE %s
// GENERIC-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "generic"

// RUN: %clang -target aarch64_be -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CA53-BE %s
// CA53-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a53"

// RUN: %clang -target aarch64_be -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a57 -### -c %s 2>&1 | FileCheck -check-prefix=CA57-BE %s
// CA57-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a57"

// RUN: %clang -target aarch64_be -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mcpu=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64_be -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64 -mbig-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// RUN: %clang -target aarch64_be -mbig-endian -mtune=cortex-a72 -### -c %s 2>&1 | FileCheck -check-prefix=CA72-BE %s
// CA72-BE: "-cc1"{{.*}} "-triple" "aarch64_be{{.*}}" "-target-cpu" "cortex-a72"

// RUN: %clang -target aarch64 -mcpu=cortex-a57 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=cortex-a57  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE %s
// RUN: %clang -target aarch64 -mcpu=cortex-a72 -mtune=cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE %s
// RUN: %clang -target aarch64 -mtune=cortex-a53 -mcpu=cortex-a72  -### -c %s 2>&1 | FileCheck -check-prefix=MCPU-MTUNE %s
// MCPU-MTUNE: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a53"

// RUN: %clang -target aarch64 -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64 -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64 -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.1a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// RUN: %clang -target aarch64_be -mlittle-endian -march=armv8.1-a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV81A %s
// GENERICV81A: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "generic" "-target-feature" "+neon" "-target-feature" "+v8.1a"

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
// RUN: %clang -target aarch64 -mtune=Cortex-a53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA53 %s
// CASE-INSENSITIVE-CA53: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a53"

// RUN: %clang -target arm64 -mcpu=cortex-A53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA53 %s
// RUN: %clang -target arm64 -mtune=cortex-A53 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA53 %s
// CASE-INSENSITIVE-ARM64-CA53: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a53"

// RUN: %clang -target aarch64 -mcpu=CORTEX-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA57 %s
// RUN: %clang -target aarch64 -mtune=CORTEX-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-CA57 %s
// CASE-INSENSITIVE-CA57: "-cc1"{{.*}} "-triple" "aarch64{{.*}}" "-target-cpu" "cortex-a57"

// RUN: %clang -target arm64 -mcpu=Cortex-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA57 %s
// RUN: %clang -target arm64 -mtune=Cortex-A57 -### -c %s 2>&1 | FileCheck -check-prefix=CASE-INSENSITIVE-ARM64-CA57 %s
// CASE-INSENSITIVE-ARM64-CA57: "-cc1"{{.*}} "-triple" "arm64{{.*}}" "-target-cpu" "cortex-a57"
