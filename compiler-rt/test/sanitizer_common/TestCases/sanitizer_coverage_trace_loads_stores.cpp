// Tests -fsanitize-coverage=trace-pc,trace-loads,trace-stores
//
// REQUIRES: has_sancovcc,stable-runtime,x86_64
//
// RUN: %clangxx -O0 %s -fsanitize-coverage=trace-pc,trace-loads,trace-stores -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stdint.h>
#include <stdio.h>

extern "C" {
void __sanitizer_cov_load1(uint8_t *addr) { printf("load1: %p\n", addr); }
void __sanitizer_cov_load2(uint16_t *addr) { printf("load2: %p\n", addr); }
void __sanitizer_cov_load4(uint32_t *addr) { printf("load4: %p\n", addr); }
void __sanitizer_cov_load8(uint64_t *addr) { printf("load8: %p\n", addr); }
void __sanitizer_cov_load16(__int128 *addr) { printf("load16: %p\n", addr); }

void __sanitizer_cov_store1(uint8_t *addr) { printf("store1: %p\n", addr); }
void __sanitizer_cov_store2(uint16_t *addr) { printf("store2: %p\n", addr); }
void __sanitizer_cov_store4(uint32_t *addr) { printf("store4: %p\n", addr); }
void __sanitizer_cov_store8(uint64_t *addr) { printf("store8: %p\n", addr); }
void __sanitizer_cov_store16(__int128 *addr) { printf("store16: %p\n", addr); }
}

uint8_t var1;
uint16_t var2;
uint32_t var4;
uint64_t var8;
__int128 var16;
static volatile int sink;

int main() {
  printf("var1: %p\n", &var1);
  sink = var1;
  var1 = 42;
  // CHECK: var1: [[ADDR:0x.*]]
  // CHECK: load1: [[ADDR]]
  // CHECK: store1: [[ADDR]]

  printf("var2: %p\n", &var2);
  sink = var2;
  var2 = 42;
  // CHECK: var2: [[ADDR:0x.*]]
  // CHECK: load2: [[ADDR]]
  // CHECK: store2: [[ADDR]]

  printf("var4: %p\n", &var4);
  sink = var4;
  var4 = 42;
  // CHECK: var4: [[ADDR:0x.*]]
  // CHECK: load4: [[ADDR]]
  // CHECK: store4: [[ADDR]]

  printf("var8: %p\n", &var8);
  sink = var8;
  var8 = 42;
  // CHECK: var8: [[ADDR:0x.*]]
  // CHECK: load8: [[ADDR]]
  // CHECK: store8: [[ADDR]]

  printf("var16: %p\n", &var16);
  sink = var16;
  var16 = 42;
  // CHECK: var16: [[ADDR:0x.*]]
  // CHECK: load16: [[ADDR]]
  // CHECK: store16: [[ADDR]]
  printf("PASS\n");
}
