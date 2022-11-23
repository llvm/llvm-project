// -----------------------------------------------------------------------------
// Tests for the hvx features and warnings.
// -----------------------------------------------------------------------------

// No HVX without -mhvx/-mhvx=

// CHECK-HVX-ON:      "-target-feature" "+hvx
// CHECK-HVX-ON-NOT:  "-target-feature" "-hvx
// CHECK-HVX-OFF-NOT: "-target-feature" "+hvx

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv5 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv55 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv65 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv66 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv67 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv67t \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv68 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv71 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv71t \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv73 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s

// Infer HVX version from flag:

// CHECK-HVX-V60: "-target-feature" "+hvxv60"
// CHECK-HVX-V62: "-target-feature" "+hvxv62"
// CHECK-HVX-V65: "-target-feature" "+hvxv65"
// CHECK-HVX-V66: "-target-feature" "+hvxv66"
// CHECK-HVX-V67: "-target-feature" "+hvxv67"
// CHECK-HVX-V68: "-target-feature" "+hvxv68"
// CHECK-HVX-V69: "-target-feature" "+hvxv69"
// CHECK-HVX-V71: "-target-feature" "+hvxv71"
// CHECK-HVX-V73: "-target-feature" "+hvxv73"

// Direct version flag:
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v60 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V60 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v62 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V62 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v65 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V65 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v66 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V66 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v67 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V67 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v68 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V68 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v69 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V69 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v71 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V71 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v73 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V73 %s
// Infer HVX version from CPU version:
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V60 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V62 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv65 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V65 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv66 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V66 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv67 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V67 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv67t -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V67 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv68 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V68 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V69 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv71 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V71 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv73 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V73 %s

// Direct version flag with different CPU version:
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v60 -mv62 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V60 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v62 -mv65 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V62 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v65 -mv66 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V65 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v66 -mv67 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V66 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v67 -mv68 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V67 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v68 -mv69 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V68 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v69 -mv71 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V69 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v71 -mv73 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V71 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v73 -mv60 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V73 %s

// Direct version flag with different CPU version and versionless -mhvx:
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v60 -mv62 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V62 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v62 -mv65 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V65 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v65 -mv66 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V66 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v66 -mv67 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V67 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v67 -mv68 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V68 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v68 -mv69 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V69 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v69 -mv71 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V71 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v71 -mv73 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V73 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v73 -mv60 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V60 %s

// Direct version flag with different CPU version, versionless -mhvx
// and -mno-hvx. The -mno-hvx cancels -mhvx=, versionless -mhvx wins:
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v60 -mno-hvx -mv62 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V62 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v62 -mno-hvx -mv65 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V65 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v65 -mno-hvx -mv66 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V66 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v66 -mno-hvx -mv67 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V67 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v67 -mno-hvx -mv68 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V68 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v68 -mno-hvx -mv69 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V69 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v69 -mno-hvx -mv71 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V71 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v71 -mno-hvx -mv73 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V73 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v73 -mno-hvx -mv60 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V60 %s

// Direct version flag with different CPU version, versionless -mhvx
// and -mno-hvx. The -mno-hvx cancels versionless -mhvx, -mhvx= wins:
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx -mno-hvx -mhvx=v60 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V60 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv65 -mhvx -mno-hvx -mhvx=v62 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V62 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv66 -mhvx -mno-hvx -mhvx=v65 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V65 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv67 -mhvx -mno-hvx -mhvx=v66 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V66 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv68 -mhvx -mno-hvx -mhvx=v67 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V67 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mno-hvx -mhvx=v68 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V68 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv71 -mhvx -mno-hvx -mhvx=v69 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V69 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv73 -mhvx -mno-hvx -mhvx=v71 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V71 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx -mno-hvx -mhvx=v73 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-V73 %s

// Infer HVX length from flag:

// CHECK-HVX-L64:  "-target-feature" "+hvx-length64b"
// CHECK-HVX-L64-NOT:  "-target-feature" "+hvx-length128b"
// CHECK-HVX-L128: "-target-feature" "+hvx-length128b"
// CHECK-HVX-L128-NOT: "-target-feature" "+hvx-length64b"

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx -mhvx-length=64b \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L64 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx -mhvx-length=128b \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L128 %s

// Infer HVX length from HVX version:

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v60 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L64 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v62 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L64 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v65 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L64 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v66 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L128 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v67 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L128 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v68 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L128 %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v69 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-L128 %s

// No HVX with trailing -mno-hvx

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v69 -mno-hvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mno-hvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx=v69 -mhvx-length=128b -mno-hvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mhvx-qfloat -mno-hvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-OFF %s

// Float

// CHECK-HVX-QFLOAT-ON:      "-target-feature" "+hvx-qfloat"
// CHECK-HVX-QFLOAT-OFF-NOT: "-target-feature" "+hvx-qfloat"
// CHECK-HVX-IEEE-ON:        "-target-feature" "+hvx-ieee-fp"
// CHECK-HVX-IEEE-OFF-NOT:   "-target-feature" "+hvx-ieee-fp"

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mhvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-QFLOAT-ON %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mno-hvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-QFLOAT-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mno-hvx-qfloat -mhvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-QFLOAT-ON %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mhvx-qfloat -mno-hvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-QFLOAT-OFF %s

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mhvx-ieee-fp \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-IEEE-ON %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mno-hvx-ieee-fp \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-IEEE-OFF %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mno-hvx-ieee-fp -mhvx-ieee-fp \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-IEEE-ON %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx -mhvx-ieee-fp -mno-hvx-ieee-fp \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-IEEE-OFF %s

// HVX flags heed HVX:

// CHECK-NEEDS-HVX: error: {{.*}} requires HVX, use -mhvx/-mhvx= to enable it

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv66 -mhvx-length=64b \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NEEDS-HVX %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv66 -mhvx-length=128b \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NEEDS-HVX %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx-qfloat \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NEEDS-HVX %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv69 -mhvx-ieee-fp \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NEEDS-HVX %s

// Invalid HVX length:

// CHECK-HVX-BAD-LENGTH: error: unsupported argument '{{.*}}' to option '-mhvx-length='

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx -mhvx-length=B \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-BAD-LENGTH %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx -mhvx-length=128 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-BAD-LENGTH %s
