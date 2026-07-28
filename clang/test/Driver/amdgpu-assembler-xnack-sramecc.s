// REQUIRES: amdgpu-registered-target

/// Assembling a .s file that has no target ID directive (.amdgcn_target). The
/// xnack/sramecc mode is a property of the assembly / object, so specifying it
/// on the command line, either with -mxnack/-msramecc or via the -mcpu target
/// ID modifiers, is not preserved: the driver no longer forwards these to the
/// assembler, and the resulting object uses the "any" setting in its e_flags.
/// To pin a mode, the .s file must contain a .amdgcn_target directive.

// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -mno-xnack -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CC1AS %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -mxnack -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CC1AS %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906 -mno-sramecc -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CC1AS %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906:xnack+ -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CC1AS %s
// RUN: %clang -### --target=amdgcn-amd-amdhsa -mcpu=gfx906:sramecc- -c %s 2>&1 | \
// RUN:   FileCheck -check-prefix=CC1AS %s

/// The assembler is invoked with the plain target-cpu and no xnack/sramecc
/// target feature.
// CC1AS: "-cc1as"
// CC1AS-SAME: "-target-cpu" "gfx906"
// CC1AS-NOT: "-target-feature"

/// End to end: the mode requested on the command line is not reflected in the
/// object, which keeps the "any" xnack/sramecc e_flags.
// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx906 -mno-xnack -c %s -o %t.o 2>&1 | \
// RUN:   FileCheck -check-prefix=UNUSED %s
// RUN: llvm-readobj --file-headers %t.o | FileCheck -check-prefix=ANY %s

// RUN: %clang --target=amdgcn-amd-amdhsa -mcpu=gfx906:xnack+ -c %s -o %t.o
// RUN: llvm-readobj --file-headers %t.o | FileCheck -check-prefix=ANY %s

// UNUSED: warning: argument unused during compilation: '-mno-xnack'
// ANY: EF_AMDGPU_FEATURE_SRAMECC_ANY_V4 (0x400)
// ANY: EF_AMDGPU_FEATURE_XNACK_ANY_V4 (0x100)
