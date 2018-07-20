// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: env CINDEXTEST_INVOCATION_EMISSION_PATH=%t not c-index-test -code-completion-at=%s:10:1 "-remap-file=%s,%S/Inputs/record-parsing-invocation-remap.c" %s
// RUN: %clang -cc1gen-reproducer %t/libclang-* -v | FileCheck %s

// Invocation file must be removed by clang:
// RUN: ls %t | count 0

// CHECK: REPRODUCER METAINFO: {"libclang.operation": "complete", "libclang.opts": "1", "invocation-args": ["-code-completion-at={{.*}}create-libclang-completion-reproducer.c:10:1"]}

// CHECK: REPRODUCER:
// CHECK-NEXT: {
// CHECK-NEXT: "files":["{{.*}}.c","{{.*}}.sh"]
// CHECK-NEXT: }
