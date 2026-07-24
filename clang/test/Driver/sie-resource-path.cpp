// REQUIRES: clang-vendor=SIE

// Test relative CLANG_RESOURCE_PATH configuration, as used on SIE toolchains.
// --target shouldn't have an impact on this as it's a build config.

// RUN: %clang -c -### %s 2>&1 | FileCheck %s
// Expected resource path.
// CHECK: "-resource-dir" "{{.*}}{{/|\\\\}}lib{{/|\\\\}}clang"
// Check resource path doesn't have a version number at the end.
// CHECK-NOT: "-resource-dir" "{{.*}}{{/|\\\\}}lib{{/|\\\\}}clang{{/|\\\\}}{{[0-9]}}"
// Check resource path doesn't have .. before lib.
// CHECK-NOT: "-resource-dir" "{{.*}}..{{/|\\\\}}lib{{/|\\\\}}clang"
