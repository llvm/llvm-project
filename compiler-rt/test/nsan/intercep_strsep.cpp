// RUN: %clangxx_nsan -O2 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <cstring>
#include <iostream>

extern "C" void __nsan_dump_shadow_mem(const char *addr, size_t size_bytes,
                                       size_t bytes_per_line, size_t reserved);

int main() {
  // Define a C-style string with commas as delimiters
  char input[] = "apple,banana,cherry,date";
  char *token;
  char *rest = input; // Pointer to keep track of the rest of the string

  // Tokenize the string using strsep
  while ((token = strsep(&rest, ",")) != NULL) {
    std::cout << token << std::endl;
  }

  __nsan_dump_shadow_mem(&input[5], 1, 1, 0);
  // CHECK: 0x{{[a-f0-9]*}}:    _
  return 0;
}
