// RUN: %clang_cc1 -E -dM -triple aarch64-unknown-linux -fexperimental-pointer-field-protection-abi %s | FileCheck %s --check-prefix=PFP
// RUN: %clang_cc1 -E -dM -triple aarch64-unknown-linux -fexperimental-pointer-field-protection-abi -fexperimental-pointer-field-protection-tagged %s | FileCheck %s --check-prefixes=PFP,PFP-TAGGED

// PFP: #define __POINTER_FIELD_PROTECTION_ABI__ 1
// PFP-TAGGED: #define __POINTER_FIELD_PROTECTION_TAGGED__ 1
