/// Check that IGCA target macros are predefined correctly during SYCL device
/// compilation if an IGCA target was provided through -target-cpu.

/// Core feature set:
// RUN: %clang_cc1 %s -fsycl-is-device -target-cpu igca_20 -E -dM \
// RUN:   | FileCheck %s --check-prefix=CORE --implicit-check-not=__OFFLOAD_ARCH_IGCA
// CORE: #define __OFFLOAD_ARCH_IGCA__ 20

/// Compute / render feature sets:
// RUN: %clang_cc1 %s -fsycl-is-device -target-cpu igca_20c -E -dM \
// RUN:   | FileCheck %s --check-prefix=COMPUTE --implicit-check-not=__OFFLOAD_ARCH_IGCA
// COMPUTE-DAG: #define __OFFLOAD_ARCH_IGCA__ 20
// COMPUTE-DAG: #define __OFFLOAD_ARCH_IGCA_COMPUTE__ 1

// RUN: %clang_cc1 %s -fsycl-is-device -target-cpu igca_15r -E -dM \
// RUN:   | FileCheck %s --check-prefix=RENDER --implicit-check-not=__OFFLOAD_ARCH_IGCA
// RENDER-DAG: #define __OFFLOAD_ARCH_IGCA__ 15
// RENDER-DAG: #define __OFFLOAD_ARCH_IGCA_RENDER__ 1

/// Compute-exact / render-exact:
// RUN: %clang_cc1 %s -fsycl-is-device -target-cpu igca_20ca -E -dM \
// RUN:   | FileCheck %s --check-prefix=COMPUTE-EXACT --implicit-check-not=__OFFLOAD_ARCH_IGCA
// COMPUTE-EXACT-DAG: #define __OFFLOAD_ARCH_IGCA__ 20
// COMPUTE-EXACT-DAG: #define __OFFLOAD_ARCH_IGCA_COMPUTE__ 1
// COMPUTE-EXACT-DAG: #define __OFFLOAD_ARCH_IGCA_COMPUTE_EXACT__ 1

// RUN: %clang_cc1 %s -fsycl-is-device -target-cpu igca_15ra -E -dM \
// RUN:   | FileCheck %s --check-prefix=RENDER-EXACT --implicit-check-not=__OFFLOAD_ARCH_IGCA
// RENDER-EXACT-DAG: #define __OFFLOAD_ARCH_IGCA__ 15
// RENDER-EXACT-DAG: #define __OFFLOAD_ARCH_IGCA_RENDER__ 1
// RENDER-EXACT-DAG: #define __OFFLOAD_ARCH_IGCA_RENDER_EXACT__ 1

/// No IGCA macros should be defined without an IGCA target:
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM \
// RUN:   | FileCheck %s --check-prefix=NONE --implicit-check-not=__OFFLOAD_ARCH_IGCA
/// IGCA macros should only be defined during device compilation:
// RUN: %clang_cc1 %s -triple spirv64-unknown-unknown -target-cpu igca_20ca -E -dM \
// RUN:   | FileCheck %s --check-prefix=NONE --implicit-check-not=__OFFLOAD_ARCH_IGCA
// NONE: #define __STDC__ 1

/// Reject unknown IGCA targets:
// RUN: not %clang_cc1 %s -fsycl-is-device -target-cpu igca_42 -E -dM 2>&1 \
// RUN:   | FileCheck %s --check-prefix=INVALID
// INVALID: error: unknown target CPU 'igca_42'
