// RUN: %clangxx -O1 %s -o %t && %env_tool_opts=handle_sigtrap=2 not %run %t 2>&1 | FileCheck %s

int main() {
  __builtin_debugtrap();
}

// CHECK: Sanitizer:DEADLYSIGNAL
// CHECK: Sanitizer: TRAP on unknown address
