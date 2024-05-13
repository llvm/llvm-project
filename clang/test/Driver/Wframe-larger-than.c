// RUN: %clang -Wframe-larger-than=42 \
// RUN:   -v -E %s 2>&1 | FileCheck %s --check-prefix=ENABLE
// RUN: %clang -Wframe-larger-than=42 -Wno-frame-larger-than \
// RUN:   -v -E %s 2>&1 | FileCheck %s --check-prefix=DISABLE
// RUN: %clang -Wframe-larger-than=42 -Wno-frame-larger-than -Wframe-larger-than=43 \
// RUN:   -v -E %s 2>&1 | FileCheck %s --check-prefix=REENABLE
// RUN: not %clang -Wframe-larger-than= \
// RUN:   -v -E %s 2>&1 | FileCheck %s --check-prefix=NOARG
// RUN: not %clang -Wframe-larger-than \
// RUN:   -v -E %s 2>&1 | FileCheck %s --check-prefix=NOARG
// RUN: not %clang -Wframe-larger-than=0x2a -E %s 2>&1 | FileCheck %s --check-prefix=INVALID

// ENABLE: cc1 {{.*}} -fwarn-stack-size=42 {{.*}} -Wframe-larger-than=42
// ENABLE: frame-larger-than:
// ENABLE-SAME: warning

// DISABLE: cc1 {{.*}} -fwarn-stack-size=42 {{.*}} -Wframe-larger-than=42 -Wno-frame-larger-than
// DISABLE: frame-larger-than:
// DISABLE-SAME: ignored

// REENABLE: cc1 {{.*}} -fwarn-stack-size=43 {{.*}} -Wframe-larger-than=42 -Wno-frame-larger-than -Wframe-larger-than=43
// REENABLE: frame-larger-than:
// REENABLE-SAME: warning

// NOARG: error: invalid argument '' to -Wframe-larger-than=
// INVALID: error: invalid argument '0x2a' to -Wframe-larger-than=

// We need to create some state transitions before the pragma will dump anything.
#pragma clang diagnostic push
#pragma clang diagnostic warning "-Wframe-larger-than"
#pragma clang diagnostic pop

#pragma clang __debug diag_mapping "frame-larger-than"
