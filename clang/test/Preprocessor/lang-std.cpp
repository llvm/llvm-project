// UNSUPPORTED: default-std-cxx, ps4, ps5
/// Test default standards when CLANG_DEFAULT_STD_CXX is unspecified.
/// PS4/PS5 default to gnu++14.

// RUN: %clang_cc1 -dM -E %s | FileCheck --check-prefix=CXX17 %s
// RUN: %clang_cc1 -dM -E -x cuda %s | FileCheck --check-prefix=CXX14 %s
// RUN: %clang_cc1 -dM -E -x hip %s | FileCheck --check-prefix=CXX14 %s

// RUN: %clang_cc1 -dM -E -x cuda -std=c++14 %s | FileCheck --check-prefix=CXX14 %s
// RUN: %clang_cc1 -dM -E -x hip -std=c++98 %s | FileCheck --check-prefix=CXX98 %s

// CXX98: #define __cplusplus 199711L
// CXX14: #define __cplusplus 201402L
// CXX17: #define __cplusplus 201703L
