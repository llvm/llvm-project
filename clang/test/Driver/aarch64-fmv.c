// Test which driver flags enable/disable Function Multiversioning on aarch64.

// FMV is enabled for non-android aarch64 targets:
// RUN: %clang --target=aarch64 --rtlib=compiler-rt -### -c %s 2>&1 | FileCheck -check-prefix=FMV-ENABLED %s
// RUN: %clang --target=aarch64-linux-gnu --rtlib=compiler-rt -### -c %s 2>&1 | FileCheck -check-prefix=FMV-ENABLED %s
// RUN: %clang --target=arm64-apple-ios --rtlib=compiler-rt -### -c %s 2>&1 | FileCheck -check-prefix=FMV-ENABLED %s
// RUN: %clang --target=arm64-apple-macosx --rtlib=compiler-rt -### -c %s 2>&1 | FileCheck -check-prefix=FMV-ENABLED %s

// android23 defaults to --rtlib=compiler-rt:
// RUN: %clang --target=aarch64-linux-android23 -### -c %s 2>&1 | FileCheck -check-prefix=FMV-ENABLED %s
// RUN: %clang --target=aarch64-linux-android23 --rtlib=compiler-rt  -### -c %s 2>&1 | FileCheck -check-prefix=FMV-ENABLED %s

// FMV is disabled without compiler-rt:
// RUN: %clang --rtlib=libgcc --target=aarch64 -### -c %s 2>&1 | FileCheck -check-prefix=FMV-DISABLED %s
// RUN: %clang --rtlib=libgcc --target=aarch64-linux-gnu -### -c %s 2>&1 | FileCheck -check-prefix=FMV-DISABLED %s

// Disabled for older android versions:
// RUN: %clang --rtlib=compiler-rt --target=aarch64-linux-android               -### -c %s 2>&1 | FileCheck -check-prefix=FMV-DISABLED %s
// RUN: %clang --rtlib=compiler-rt --target=aarch64-linux-android22             -### -c %s 2>&1 | FileCheck -check-prefix=FMV-DISABLED %s
// RUN: %clang --rtlib=compiler-rt --target=aarch64-linux-android22  -mno-fmv   -### -c %s 2>&1 | FileCheck -check-prefix=FMV-DISABLED %s

// Disabled explicitly:
// RUN: %clang --rtlib=compiler-rt --target=aarch64 -mno-fmv -### -c %s 2>&1 | FileCheck -check-prefix=FMV-DISABLED %s
// RUN: %clang --rtlib=compiler-rt --target=aarch64-linux-android23  -mno-fmv   -### -c %s 2>&1 | FileCheck -check-prefix=FMV-DISABLED %s

// FMV-ENABLED-NOT: "-target-feature" "-fmv"
// FMV-DISABLED: "-target-feature" "-fmv"
