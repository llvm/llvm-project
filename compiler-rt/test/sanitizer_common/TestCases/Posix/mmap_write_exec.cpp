// RUN: %clangxx %s -o %t
// RUN: %env_tool_opts=detect_write_exec=1 %run %t 2>&1 | FileCheck %s
// ubsan and lsan do not install mmap interceptors
// UNSUPPORTED: ubsan, lsan

#include <sys/mman.h>

int main(int argc, char **argv) {
  char *p = (char *)mmap(0, 1024, PROT_READ | PROT_WRITE | PROT_EXEC,
                         MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
  // CHECK: WARNING: {{.*}}Sanitizer: writable-executable page usage
}
