// RUN: %asan_symbolize --help > %t_help_output.txt
// RUN: FileCheck %s -input-file=%t_help_output.txt
// CHECK: optional arguments:
// CHECK: --log-dest
// CHECK: --log-level
