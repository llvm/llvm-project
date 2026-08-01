// REQUIRES: amdgpu-registered-target

/// Assembling a .s file that has no target ID directive (.amdgcn_target). The
/// xnack/sramecc mode requested on the command line, either with
/// -mxnack/-msramecc or via the -mcpu target ID modifiers, is forwarded to the
/// assembler as a target feature and recorded in the object's e_flags.

/// The mode is forwarded to the assembler as a target feature.
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -mno-xnack -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=XNACK-OFF %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906:xnack- -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=XNACK-OFF %s
// XNACK-OFF: "-cc1as"
// XNACK-OFF-SAME: "-target-feature" "-xnack"

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -msramecc -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SRAMECC-ON %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906:sramecc+ -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=SRAMECC-ON %s
// SRAMECC-ON: "-cc1as"
// SRAMECC-ON-SAME: "-target-feature" "+sramecc"

/// End to end: the requested mode is reflected in the object's e_flags.
// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx906 -mno-xnack -c %s -o %t.o
// RUN: llvm-readobj --file-headers %t.o | FileCheck -check-prefix=OBJ-XNACK-OFF %s
// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx906:xnack+ -c %s -o %t.o
// RUN: llvm-readobj --file-headers %t.o | FileCheck -check-prefix=OBJ-XNACK-ON %s
// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx906:sramecc- -c %s -o %t.o
// RUN: llvm-readobj --file-headers %t.o | FileCheck -check-prefix=OBJ-SRAMECC-OFF %s

// OBJ-XNACK-OFF: EF_AMDGPU_FEATURE_XNACK_OFF_V4 (0x200)
// OBJ-XNACK-ON: EF_AMDGPU_FEATURE_XNACK_ON_V4 (0x300)
// OBJ-SRAMECC-OFF: EF_AMDGPU_FEATURE_SRAMECC_OFF_V4 (0x800)
