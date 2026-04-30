// RUN: %clang_cc1 -E -dM -x cuda %s \
// RUN:   | FileCheck -match-full-lines %s --check-prefix=NO-WRAPPER
// RUN: %clang_cc1 -E -dM -x cuda \
// RUN:   -include __clang_cuda_runtime_wrapper.h \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -internal-isystem %S/Inputs/include \
// RUN:   %s | FileCheck -match-full-lines %s --check-prefix=WRAPPER

// NO-WRAPPER-NOT: #define _CG_CLUSTER_INTRINSICS_AVAILABLE 1
// WRAPPER: #define _CG_CLUSTER_INTRINSICS_AVAILABLE 1
