// RUN: export LSAN_OPTIONS=detect_leaks=0
// RUN: rm -rf %t
// RUN: not %crash_opt env CLANG_CRASH_DIAGNOSTICS_DIR=%t %clang -c %s -o - 2>&1 | FileCheck %s
#pragma clang __debug parser_crash
// CHECK: PLEASE ATTACH THE FOLLOWING CRASH REPRODUCER FILES TO THE BUG REPORT:
// CHECK: diagnostic msg: {{.*}}{{/|\\}}crash-diagnostics-dir-3.c.tmp{{(/|\\).*}}.c
