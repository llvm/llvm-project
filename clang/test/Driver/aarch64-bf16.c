// The BFloat16 extension is a mandatory component of the Armv8.6-A extensions, but is permitted as an
// optional feature for any implementation of Armv8.2-A to Armv8.5-A (inclusive)
// RUN: %clang --target=aarch64 -march=armv8.5a+bf16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BF16 %s
// GENERICV85A-BF16: "-target-feature" "+bf16"
// RUN: %clang --target=aarch64 -march=armv8.5a+bf16+nobf16 -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV85A-BF16-NO-BF16 %s
// GENERICV85A-BF16-NO-BF16: "-target-feature" "-bf16"
// RUN: %clang --target=aarch64 -march=armv8.5a+bf16+sve -### -c %s 2>&1 | FileCheck -check-prefixes=GENERICV85A-BF16-SVE %s
// GENERICV85A-BF16-SVE: "-target-feature" "+bf16"{{.*}} "-target-feature" "+sve"
