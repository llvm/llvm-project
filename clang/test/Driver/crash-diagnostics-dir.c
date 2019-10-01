// RUN: rm -rf %t
// RUN: not %clang -fcrash-diagnostics-dir=%t -c %s -o - 2>&1 | FileCheck %s
#pragma clang __debug parser_crash
// CHECK: Preprocessed source(s) and associated run script(s) are located at:
// CHECK: diagnostic msg: {{.*}}{{/|\\}}crash-diagnostics-dir.c.tmp{{(/|\\).*}}.c
