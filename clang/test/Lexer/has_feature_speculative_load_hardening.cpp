// RUN: %clang -E -mspeculative-load-hardening %s -o - | FileCheck --check-prefix=CHECK-SLH %s
// RUN: %clang -E -mno-speculative-load-hardening %s -o - | FileCheck --check-prefix=CHECK-NOSLH %s
// RUN: %clang -E %s -o - | FileCheck --check-prefix=CHECK-DEFAULT %s

#if __has_feature(speculative_load_hardening)
int SpeculativeLoadHardeningEnabled();
#else
int SpeculativeLoadHardeningDisabled();
#endif

// CHECK-SLH: SpeculativeLoadHardeningEnabled

// CHECK-NOSLH: SpeculativeLoadHardeningDisabled

// CHECK-DEFAULT: SpeculativeLoadHardeningDisabled
