/// Test default standards.

// RUN: %clang_cc1 -dM -E %s | grep __cplusplus >%T-cpp-std.txt
// RUN: FileCheck --input-file %T-cpp-std.txt --check-prefix=CXX17 %s

/// Check that CUDA/HIP uses the same default standards as C++.

// RUN: %clang_cc1 -dM -E -x cuda %s | grep __cplusplus >%T-cuda-std.txt
// RUN: %clang_cc1 -dM -E -x hip %s | grep __cplusplus >%T-hip-std.txt
// RUN: diff %T-cpp-std.txt %T-cuda-std.txt
// RUN: diff %T-cpp-std.txt %T-hip-std.txt

// RUN: %clang_cc1 -dM -E -x cuda -std=c++14 %s | FileCheck --check-prefix=CXX14 %s
// RUN: %clang_cc1 -dM -E -x cuda -std=c++17 %s | FileCheck --check-prefix=CXX17 %s
// RUN: %clang_cc1 -dM -E -x hip -std=c++98 %s | FileCheck --check-prefix=CXX98 %s

// CXX98: #define __cplusplus 199711L
// CXX14: #define __cplusplus 201402L
// CXX17: #define __cplusplus 201703L
