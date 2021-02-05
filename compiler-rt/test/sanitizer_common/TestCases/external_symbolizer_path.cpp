// REQUIRES: shell
// REQUIRES: x86-target-arch
// RUN: rm -rf %t.bin
// RUN: mkdir %t.bin
// RUN: cp $(which llvm-symbolizer) %t.bin
// RUN: rm -rf %t.dir
// RUN: mkdir %t.dir
// RUN: %clangxx -O0 %s -o %t && cd %t.dir
// RUN: %env_tool_opts=external_symbolizer_path=%d/external_symbolizer_path.cpp.tmp.bin/llvm-symbolizer \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefix=FOUND
// RUN: rm -rf %t.bin/llvm-symbolizer
// RUN: cd ..
// RUN: %clangxx -O0 %s -o %t && cd %t.dir
// RUN: %env_tool_opts=external_symbolizer_path=%d/external_symbolizer_path.cpp.tmp.bin/llvm-symbolizer \
// RUN:   %run %t 2>&1 | FileCheck %s --check-prefix=NOT-FOUND

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>

static void Symbolize() {
  char buffer[100];
  __sanitizer_symbolize_pc(__builtin_return_address(0), "%p %F %L", buffer,
                           sizeof(buffer));
  printf("%s\n", buffer);
}

int main() {
  // FOUND: {{0x.* in main}}
  // NOT-FOUND: WARNING: invalid path to external symbolizer!
  Symbolize();
}
