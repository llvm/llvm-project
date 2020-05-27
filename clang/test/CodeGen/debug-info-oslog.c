// RUN: %clang_cc1 -triple x86_64-darwin-apple -debug-info-kind=limited \
// RUN:   %s -emit-llvm -o -  | FileCheck %s
void test_builtin_os_log(void *buf, int i, const char *data) {
  __builtin_os_log_format(buf, "%d", i);
}

// CHECK: define linkonce_odr {{.*}}@__os_log_helper_1_0_1_4_0(
// CHECK-SAME:   !dbg ![[OS_LOG_HELPER:[0-9]+]]

// This helper is going to be uniqued, so it should not have a line
// number between file and type.

// CHECK: distinct !DISubprogram(name: "__os_log_helper_1_0_1_4_0",
// CHECK-SAME:                   file: !{{[0-9+]}}, type
// CHECK-SAME:                   flags: DIFlagArtificial
