// RUN: rm -rf %t
// RUN: not env CLANG_CRASH_DIAGNOSTICS_DIR=%t %clang -c %s -o - 2>&1 | FileCheck %s
#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK: diagnostic msg: {{.*}}{{/|\\}}crash-diagnostics-dir-3.c.tmp{{(/|\\).*}}.c
