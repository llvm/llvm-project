// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env CINDEXTEST_INVOCATION_EMISSION_PATH=%t not c-index-test -test-load-source all %s
// RUN: %clang -cc1gen-reproducer %t/libclang-* -v | FileCheck %s

// Invocation file must be removed by clang:
// RUN: ls %t | count 0

// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env CINDEXTEST_INVOCATION_EMISSION_PATH=%t not c-index-test -test-load-source all "-remap-file=%s,%S/Inputs/record-parsing-invocation-remap.c" %s
// RUN: %clang -cc1gen-reproducer %t/libclang-* -v | FileCheck %s

#pragma clang __debug parser_crash

// CHECK: REPRODUCER METAINFO: {"libclang.operation": "parse", "libclang.opts": "1"}

// CHECK: REPRODUCER:
// CHECK-NEXT: {
// CHECK-NEXT: "files":["{{.*}}.c","{{.*}}.sh"]
// CHECK-NEXT: }
