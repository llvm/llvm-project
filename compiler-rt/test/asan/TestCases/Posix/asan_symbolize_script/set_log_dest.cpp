// RUN: %asan_symbolize --log-level debug --log-dest %t_debug_log_output.txt --help
// RUN: FileCheck %s -input-file=%t_debug_log_output.txt -check-prefix=DEBUG-CHECK
// DEBUG-CHECK: DEBUG: [setup_logging() asan_symbolize.py:{{[0-9]+}}] Logging level set to "debug"
