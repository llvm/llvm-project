// RUN: %clangxx_memprof %s -o %t-default
// RUN: %run %t-default | FileCheck %s --check-prefix=DEFAULT

// RUN: %clangxx_memprof %s -mllvm -memprof-runtime-default-options="print_text=true,log_path=stdout,atexit=false" -o %t
// RUN: %run %t | FileCheck %s

#include <sanitizer/memprof_interface.h>
#include <stdio.h>

int main() {
  printf("Options: \"%s\"\n", __memprof_default_options());
  return 0;
}

// DEFAULT: Options: ""
// CHECK: Options: "print_text=true,log_path=stdout,atexit=false"
