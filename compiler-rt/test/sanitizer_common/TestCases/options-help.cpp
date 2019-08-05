// RUN: %clangxx -O0 %s -o %t
// RUN: %env_tool_opts=help=1 %run %t 2>&1 | FileCheck %s

int main() {
}

// CHECK: Available flags for {{.*}}Sanitizer:
// CHECK: handle_segv
