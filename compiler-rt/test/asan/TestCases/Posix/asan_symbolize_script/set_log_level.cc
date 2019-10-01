// RUN: %asan_symbolize --log-level debug --help 2> %t_debug_log_output.txt
// RUN: FileCheck %s -input-file=%t_debug_log_output.txt -check-prefix=DEBUG-CHECK
// DEBUG-CHECK: DEBUG: [setup_logging() asan_symbolize.py:{{[0-9]+}}] Logging level set to "debug"
//
// FileCheck doesn't like empty files so add stdout too.
// RUN: %asan_symbolize --log-level info --help > %t_info_log_output.txt 2>&1
// RUN: FileCheck %s -input-file=%t_info_log_output.txt -check-prefix=INFO-CHECK
// INFO-CHECK-NOT: DEBUG: [setup_logging() asan_symbolize.py:{{[0-9]+}}]
