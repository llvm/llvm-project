// The SVE extension is an optional extension for Armv8-A.
// RUN: %clang --target=aarch64 -march=armv8a+sve -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-SVE %s
// RUN: %clang --target=aarch64 -march=armv8.6a+sve -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-SVE %s
// GENERICV8A-SVE: "-target-feature" "+sve"
// RUN: %clang --target=aarch64 -march=armv8a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NOSVE %s
// RUN: %clang --target=aarch64 -march=armv8.6a -### -c %s 2>&1 | FileCheck -check-prefix=GENERICV8A-NOSVE %s
// GENERICV8A-NOSVE-NOT: "-target-feature" "+sve"

// The 32-bit floating point matrix multiply extension is enabled by default
// for armv8.6-a targets (or later) with SVE, and can optionally be enabled for
// any target from armv8.2a onwards (we don't enforce not using it with earlier
// targets).
// RUN: %clang --target=aarch64 -march=armv8.6a       -### -c %s 2>&1 | FileCheck -check-prefix=NO-F32MM %s
// RUN: %clang --target=aarch64 -march=armv8.6a+sve   -### -c %s 2>&1 | FileCheck -check-prefix=F32MM %s
// RUN: %clang --target=aarch64 -march=armv8.5a+f32mm -### -c %s 2>&1 | FileCheck -check-prefix=F32MM %s
// NO-F32MM-NOT: "-target-feature" "+f32mm"
// F32MM: "-target-feature" "+f32mm"

// The 64-bit floating point matrix multiply extension is not currently enabled
// by default for any targets, because it requires an SVE vector length >= 256
// bits. When we add a CPU which has that, then it can be enabled by default,
// but for now it can only be used by adding the +f64mm feature.
// RUN: %clang --target=aarch64 -march=armv8.6a       -### -c %s 2>&1 | FileCheck -check-prefix=NO-F64MM %s
// RUN: %clang --target=aarch64 -march=armv8.6a+sve   -### -c %s 2>&1 | FileCheck -check-prefix=NO-F64MM %s
// RUN: %clang --target=aarch64 -march=armv8.6a+f64mm -### -c %s 2>&1 | FileCheck -check-prefix=F64MM %s
// NO-F64MM-NOT: "-target-feature" "+f64mm"
// F64MM: "-target-feature" "+f64mm"
