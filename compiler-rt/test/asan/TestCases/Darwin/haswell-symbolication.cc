// RUN: %clangxx_asan -arch x86_64 -arch x86_64h -g -O0 %s -c -o %t.o.fat
// RUN: %clangxx_asan -arch x86_64 -arch x86_64h -g %t.o.fat -o %t.fat

// RUN: lipo %t.fat -thin    x86_64  -output %t.thin.x86_64  && lipo %t.thin.x86_64  -info | FileCheck %s --check-prefix=CHECK-LIPO-THIN-X86_64
// RUN: lipo %t.fat -thin    x86_64h -output %t.thin.x86_64h && lipo %t.thin.x86_64h -info | FileCheck %s --check-prefix=CHECK-LIPO-THIN-X86_64H
// RUN: lipo %t.fat -extract x86_64  -output %t.fat.x86_64   && lipo %t.fat.x86_64   -info | FileCheck %s --check-prefix=CHECK-LIPO-FAT-X86_64
// RUN: lipo %t.fat -extract x86_64h -output %t.fat.x86_64h  && lipo %t.fat.x86_64h  -info | FileCheck %s --check-prefix=CHECK-LIPO-FAT-X86_64H

// CHECK-LIPO-THIN-X86_64:  Non-fat file: {{.*}} is architecture: x86_64
// CHECK-LIPO-THIN-X86_64H: Non-fat file: {{.*}} is architecture: x86_64h
// CHECK-LIPO-FAT-X86_64:   Architectures in the fat file: {{.*}} are: x86_64
// CHECK-LIPO-FAT-X86_64H:  Architectures in the fat file: {{.*}} are: x86_64h

// RUN: dsymutil %t.thin.x86_64
// RUN: dsymutil %t.thin.x86_64h
// RUN: dsymutil %t.fat.x86_64
// RUN: dsymutil %t.fat.x86_64h

// Check LLVM symbolizer
// RUN: %env_asan_opts=external_symbolizer_path=$(which llvm-symbolizer) not %run %t.thin.x86_64  2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path=$(which llvm-symbolizer) not %run %t.thin.x86_64h 2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path=$(which llvm-symbolizer) not %run %t.fat.x86_64   2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path=$(which llvm-symbolizer) not %run %t.fat.x86_64h  2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA

// Check atos
// RUN: %env_asan_opts=external_symbolizer_path=$(which atos) not %run %t.thin.x86_64  2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path=$(which atos) not %run %t.thin.x86_64h 2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path=$(which atos) not %run %t.fat.x86_64   2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path=$(which atos) not %run %t.fat.x86_64h  2>&1 | FileCheck %s --check-prefixes CHECK,CHECK-LI,CHECK-DATA

// Check dladdr
// RUN: %env_asan_opts=external_symbolizer_path= not %run %t.thin.x86_64  2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOLI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path= not %run %t.thin.x86_64h 2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOLI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path= not %run %t.fat.x86_64   2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOLI,CHECK-DATA
// RUN: %env_asan_opts=external_symbolizer_path= not %run %t.fat.x86_64h  2>&1 | FileCheck %s --check-prefixes=CHECK,CHECK-NOLI,CHECK-DATA

// Check asan_symbolize.py with llvm-symbolizer
// RUN: %env_asan_opts=symbolize=0 not %run %t.thin.x86_64  2>&1 | %asan_symbolize | FileCheck %s --check-prefixes CHECK,CHECK-LI
// RUN: %env_asan_opts=symbolize=0 not %run %t.thin.x86_64h 2>&1 | %asan_symbolize | FileCheck %s --check-prefixes CHECK,CHECK-LI
// RUN: %env_asan_opts=symbolize=0 not %run %t.fat.x86_64   2>&1 | %asan_symbolize | FileCheck %s --check-prefixes CHECK,CHECK-LI
// RUN: %env_asan_opts=symbolize=0 not %run %t.fat.x86_64h  2>&1 | %asan_symbolize | FileCheck %s --check-prefixes CHECK,CHECK-LI

// Check asan_symbolize.py with atos
// RUN: %env_asan_opts=symbolize=0 not %run %t.thin.x86_64  2>&1 | %asan_symbolize --force-system-symbolizer | FileCheck %s --check-prefixes CHECK,CHECK-LI
// RUN: %env_asan_opts=symbolize=0 not %run %t.thin.x86_64h 2>&1 | %asan_symbolize --force-system-symbolizer | FileCheck %s --check-prefixes CHECK,CHECK-LI
// RUN: %env_asan_opts=symbolize=0 not %run %t.fat.x86_64   2>&1 | %asan_symbolize --force-system-symbolizer | FileCheck %s --check-prefixes CHECK,CHECK-LI
// RUN: %env_asan_opts=symbolize=0 not %run %t.fat.x86_64h  2>&1 | %asan_symbolize --force-system-symbolizer | FileCheck %s --check-prefixes CHECK,CHECK-LI

// REQUIRES: x86-target-arch
// REQUIRES: x86_64h

#include <sanitizer/common_interface_defs.h>
#include <stdio.h>
#include <stdlib.h>

#if __x86_64h__
// Unused functions and globals, just to mess up the offsets in x86_64h.
void dummy(char *a, char *b) {
  while (*a == *b) {
    a[0] = b[0];
    a[1] = b[1];
    a[2] = b[2];
    a[3] = b[3];
    fprintf(stderr, "dummy\n");
  }
  fprintf(stderr, "dummy\n");
}
long dummy_global;
long dummy_global2[100];
#endif

extern "C"
long faulty_global = 10;

void check_data_symbolication() {
  char data[100];
  __sanitizer_symbolize_global(&faulty_global, "%g", data, sizeof(data));
  fprintf(stderr, "symbolized global: %s\n", data);
  // CHECK-DATA: symbolized global: faulty_global
}

extern "C"
void faulty_func(char *p) {
  *p = 'x';  // BOOM
  // CHECK: AddressSanitizer: global-buffer-overflow
  // CHECK-LI: #0 0x{{.*}} in faulty_func{{.*}} {{.*}}haswell-symbolication.cc:[[@LINE-2]]
  // CHECK-NOLI: #0 0x{{.*}} in faulty_func{{.*}} {{.*}}haswell-symbolication
  // CHECK: is located 2 bytes to the right of global variable 'faulty_global'
  // CHECK-NOT: LLVMSymbolizer: error reading file
}

int main() {
  check_data_symbolication();

  char *p = (char *)(void *)&faulty_global;
  p += 10;
  faulty_func(p);
  return 0;
}
