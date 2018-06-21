// -----------------------------------------------------------------------------
// Tests for the hvx features and warnings.
// -----------------------------------------------------------------------------

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv65 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECKHVX165 %s
// CHECKHVX165: "-target-feature" "+hvxv65"

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECKHVX162 %s
// CHECKHVX162: "-target-feature" "+hvxv62"

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv65 -mhvx \
// RUN:  -mhvx-double 2>&1 | FileCheck -check-prefix=CHECKHVX2 %s

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx \
// RUN:  -mhvx-double 2>&1 | FileCheck -check-prefix=CHECKHVX2 %s

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv65 -mhvx \
// RUN:  -mhvx-length=128B 2>&1 | FileCheck -check-prefix=CHECKHVX2 %s

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx \
// RUN:  -mhvx-length=128B 2>&1 | FileCheck -check-prefix=CHECKHVX2 %s

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx \
// RUN:  -mhvx-length=128b 2>&1 | FileCheck -check-prefix=CHECKHVX2 %s
// CHECKHVX2-NOT: "-target-feature" "+hvx-length64b"
// CHECKHVX2: "-target-feature" "+hvx-length128b"

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv65 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECKHVX3 %s

// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 2>&1 \
// RUN:  | FileCheck -check-prefix=CHECKHVX3 %s
// CHECKHVX3-NOT: "-target-feature" "+hvx

// -mhvx-double is deprecated.
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx-double \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-DEPRECATED %s
// CHECK-DEPRECATED: warning: argument '-mhvx-double' is deprecated, use '-mhvx-length=128B' instead [-Wdeprecated]

// -mno-hvx-double is deprecated.
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mno-hvx-double \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NODEPRECATED %s
// CHECK-NODEPRECATED: warning: argument '-mno-hvx-double' is deprecated, use '-mno-hvx' instead [-Wdeprecated]

// No hvx target feature must be added if -mno-hvx/-mno-hvx-double occurs last
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mno-hvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NOHVX %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx -mno-hvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NOHVX %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx -mno-hvx-double \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-NOHVX %s
// CHECK-NOHVX-NOT: "-target-feature" "+hvx

// Hvx target feature should be added if -mno-hvx doesnot occur last
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mno-hvx -mhvx\
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXFEAT %s
// CHECK-HVXFEAT: "-target-feature" "+hvxv62"

// With -mhvx, the version of hvx defaults to Cpu
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVX-DEFAULT %s
// CHECK-HVX-DEFAULT: "-target-feature" "+hvxv60"

// Test -mhvx= flag
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx=v62 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXEQ %s
// CHECK-HVXEQ: "-target-feature" "+hvxv62"

// Honor the last occurred -mhvx=, -mhvx flag.
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx=v62 -mhvx\
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXEQ-PRE %s
// CHECK-HVXEQ-PRE-NOT: "-target-feature" "+hvxv62"
// CHECK-HVXEQ-PRE: "-target-feature" "+hvxv60"
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx -mhvx=v62\
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXEQ-PRE2 %s
// CHECK-HVXEQ-PRE2-NOT: "-target-feature" "+hvxv60"
// CHECK-HVXEQ-PRE2: "-target-feature" "+hvxv62"

// Test -mhvx-length flag
// The default mode on v60,v62 is 64B.
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-64B %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx \
// RUN:  -mhvx-length=64b 2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-64B %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv60 -mhvx \
// RUN:  -mhvx-length=64B 2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-64B %s
// CHECK-HVXLENGTH-64B: "-target-feature" "+hvx{{.*}}" "-target-feature" "+hvx-length64b"
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mv62 -mhvx -mhvx-length=128B\
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-128B %s
// CHECK-HVXLENGTH-128B: "-target-feature" "+hvx{{.*}}" "-target-feature" "+hvx-length128b"

// Bail out if -mhvx-length is specified without HVX enabled
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx-length=64B \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-ERROR %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx-length=128B \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-ERROR %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx-double \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-ERROR %s
// CHECK-HVXLENGTH-ERROR: error: -mhvx-length is not supported without a -mhvx/-mhvx= flag

// Error out if an unsupported value is passed to -mhvx-length.
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx -mhvx-length=B \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-VALUE-ERROR %s
// RUN: %clang -c %s -### -target hexagon-unknown-elf -mhvx -mhvx-length=128 \
// RUN:  2>&1 | FileCheck -check-prefix=CHECK-HVXLENGTH-VALUE-ERROR %s
// CHECK-HVXLENGTH-VALUE-ERROR: error: unsupported argument '{{.*}}' to option 'mhvx-length='
