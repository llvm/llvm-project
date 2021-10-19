// RUN: %asan_symbolize --help > %t_help_output.txt
// RUN: FileCheck %s -input-file=%t_help_output.txt
// CHECK: option{{al arguments|s}}:
// CHECK: --log-dest
// CHECK: --log-level
