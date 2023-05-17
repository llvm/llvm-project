/// Test that PS4/PS5 defaults to gnu++14.

// RUN: %clang_cc1 -dM -E -triple x86_64-scei-ps4 %s | FileCheck --check-prefix=CXX14 %s
// RUN: %clang_cc1 -dM -E -triple x86_64-sie-ps5 %s | FileCheck --check-prefix=CXX14 %s

// CXX14: #define __cplusplus 201402L
