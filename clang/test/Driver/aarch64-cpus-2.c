// Check target CPUs are correctly passed. Continuation of aarch64-cpus-1.c.
// NOTE: The tests are split across multiple files, to avoid excessive test
// times for large single test files.
// TODO: The files should be split up by categories, e.g. by architecture versions.

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
// GENERIC: "-cc1"{{.*}} "-triple" "aarch64{{(--)?}}"{{.*}} "-target-cpu" "generic"

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
