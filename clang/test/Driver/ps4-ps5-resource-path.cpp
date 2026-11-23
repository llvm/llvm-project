// REQUIRES: clang-vendor=SIE

// Test relative CLANG_RESOURCE_PATH=../lib/clang configuration, as used on
// SIE toolchains (PS4/PS5).
// --target shouldn't have an impact on this as it's a build config.

// RUN: %clang -c -### %s 2>&1 | FileCheck %s
// Expected resource path doesn't have a . before, or a number after.
// CHECK: "-resource-dir" "{{.*[^.]}}{{/|\\\\}}lib{{/|\\\\}}clang"
