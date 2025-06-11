// Test how -cc1 -target-feature interacts with -print-enabled-extensions.
// The current behaviour does not look correct, since dependent features are
// removed from the printed list when one of their dependencies are disabled,
// but they are actually still enabled during compilation, and then actually
// disabled for parsing assembly.

// REQUIRES: aarch64-registered-target

// Behaviour with two positive features.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +neon -target-feature +sve \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=POS_ONLY

// Negative -target-feature disables the extension but keeps any dependencies of it (FEAT_FP16).
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +neon -target-feature +sve -target-feature -sve \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=POS_NEG

// Disabling then re-enabling a feature is the same as never disabling it.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +neon -target-feature -sve -target-feature +sve \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=POS_ONLY

// Disabling then re-enabling a feature is the same as never disabling it.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +neon -target-feature +sve -target-feature -sve -target-feature +sve \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=POS_ONLY

// Only disabling it is the same as never having enabled it.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +neon \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=NEG_ONLY

// Only disabling it is the same as never having enabled it.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +neon -target-feature -sve \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=NEG_ONLY

// Disabling a dependency (after enabling the dependent) appears to disable the dependent feature.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +sve2 -target-feature -sve \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=DISABLE_DEP

// Disabling a dependency before enabling the dependent appears to have no effect.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature -sve -target-feature +sve2 \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=DISABLE_DEP2

// Disabling a dependency before enabling the dependent appears to have no effect.
// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -print-enabled-extensions \
// RUN:     -target-feature +sve2 \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=DISABLE_DEP2

// Driver --print-enabled-extensions indicates that negative -target-features disable dependent features.
// RUN: %clang --target=aarch64 -march=armv8-a+sve2 --print-enabled-extensions \
// RUN:     -Xclang -target-feature -Xclang -sve \
// RUN:     | FileCheck --strict-whitespace --implicit-check-not=FEAT_ %s --check-prefix=DISABLE_VIA_XCLANG

// However, sve2 is actually enabled in clang but disabled for MC.
// RUN: %clang --target=aarch64 -march=armv8-a+sve2 -c %s -o %t \
// RUN:     -Xclang -target-feature -Xclang -sve \
// RUN:     -Xclang -verify -Xclang -verify-ignore-unexpected=note


// POS_ONLY: Extensions enabled for the given AArch64 target
// POS_ONLY-EMPTY:
// POS_ONLY-NEXT:     Architecture Feature(s)                                Description
// POS_ONLY-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// POS_ONLY-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// POS_ONLY-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// POS_ONLY-NEXT:     FEAT_FP16                                              Enable half-precision floating-point data processing
// POS_ONLY-NEXT:     FEAT_SVE                                               Enable Scalable Vector Extension (SVE) instructions
// POS_ONLY-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension

// POS_NEG: Extensions enabled for the given AArch64 target
// POS_NEG-EMPTY:
// POS_NEG-NEXT:     Architecture Feature(s)                                Description
// POS_NEG-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// POS_NEG-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// POS_NEG-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// POS_NEG-NEXT:     FEAT_FP16                                              Enable half-precision floating-point data processing
// POS_NEG-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension

// NEG_POS: Extensions enabled for the given AArch64 target
// NEG_POS-EMPTY:
// NEG_POS-NEXT:     Architecture Feature(s)                                Description
// NEG_POS-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// NEG_POS-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// NEG_POS-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// NEG_POS-NEXT:     FEAT_FP16                                              Enable half-precision floating-point data processing
// NEG_POS-NEXT:     FEAT_SVE                                               Enable Scalable Vector Extension (SVE) instructions
// NEG_POS-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension

// NEG_ONLY: Extensions enabled for the given AArch64 target
// NEG_ONLY-EMPTY:
// NEG_ONLY-NEXT:     Architecture Feature(s)                                Description
// NEG_ONLY-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// NEG_ONLY-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// NEG_ONLY-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// NEG_ONLY-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension

// DISABLE_DEP: Extensions enabled for the given AArch64 target
// DISABLE_DEP-EMPTY: 
// DISABLE_DEP-NEXT:     Architecture Feature(s)                                Description
// DISABLE_DEP-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// DISABLE_DEP-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// DISABLE_DEP-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// DISABLE_DEP-NEXT:     FEAT_FP16                                              Enable half-precision floating-point data processing
// DISABLE_DEP-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension

// DISABLE_DEP2: Extensions enabled for the given AArch64 target
// DISABLE_DEP2-EMPTY: 
// DISABLE_DEP2-NEXT:     Architecture Feature(s)                                Description
// DISABLE_DEP2-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// DISABLE_DEP2-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// DISABLE_DEP2-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// DISABLE_DEP2-NEXT:     FEAT_FP16                                              Enable half-precision floating-point data processing
// DISABLE_DEP2-NEXT:     FEAT_SVE                                               Enable Scalable Vector Extension (SVE) instructions
// DISABLE_DEP2-NEXT:     FEAT_SVE2                                              Enable Scalable Vector Extension 2 (SVE2) instructions
// DISABLE_DEP2-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension

// DISABLE_VIA_XCLANG: Extensions enabled for the given AArch64 target
// DISABLE_VIA_XCLANG-EMPTY: 
// DISABLE_VIA_XCLANG-NEXT:     Architecture Feature(s)                                Description
// DISABLE_VIA_XCLANG-NEXT:     FEAT_AdvSIMD                                           Enable Advanced SIMD instructions
// DISABLE_VIA_XCLANG-NEXT:     FEAT_ETE                                               Enable Embedded Trace Extension
// DISABLE_VIA_XCLANG-NEXT:     FEAT_FP                                                Enable Armv8.0-A Floating Point Extensions
// DISABLE_VIA_XCLANG-NEXT:     FEAT_FP16                                              Enable half-precision floating-point data processing
// DISABLE_VIA_XCLANG-NEXT:     FEAT_TRBE                                              Enable Trace Buffer Extension

#if __ARM_FEATURE_SVE2
#warning "SVE2 is enabled"
// expected-warning@-1 {{SVE2 is enabled}}
#endif

void fn_that_requires_sve2() {
    __asm__("ldnt1sh z0.s, p0/z, [z1.s]");
    // expected-error@-1 {{instruction requires: sve2}}
}
