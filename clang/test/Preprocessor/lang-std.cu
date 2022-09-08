// RUN: %clang_cc1 -dM -E -x hip %s | FileCheck -check-prefix=CXX14 %s
// RUN: %clang_cc1 -dM -E %s | FileCheck -check-prefix=CXX14 %s
// RUN: %clang_cc1 -dM -E -std=c++98 -x hip %s | FileCheck -check-prefix=CXX98 %s
// RUN: %clang_cc1 -dM -E -std=c++98 %s | FileCheck -check-prefix=CXX98 %s

// CXX98: #define __cplusplus 199711L
// CXX14: #define __cplusplus 201402L
