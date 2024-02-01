// RUN: %clang_cc1 -triple arm64-apple-xros1 -dM -E -o - %s | FileCheck %s

// CHECK: __ENVIRONMENT_OS_VERSION_MIN_REQUIRED__ 10000
