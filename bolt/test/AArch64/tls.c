// Test checks that BOLT can process binaries with TLS relocations

__thread struct str {
  int a;
  int b;
} tbssstruct = {}, tdatastruct = {4, 2};

extern __thread struct str extstruct;

extern void processAddr(volatile void *);

int main() {
  // R_AARCH64_TLSDESC_ADR_PAGE21 and R_AARCH64_TLSDESC_LD64_LO12_NC are
  // produced for pie binaries in all cases below.

  // The R_TLSLE_ADD_TPREL_HI12 and R_TLSLE_ADD_TPREL_LO12_NC for
  // relocations in .tbss and .tdata
  processAddr(&tbssstruct.b);
  processAddr(&tdatastruct.b);

  // The R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21 and
  // R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC relocations
  processAddr(&extstruct.b);
}

// REQUIRES: system-linux
// RUN: %clang %cflags -no-pie %s -o %t.exe -Wl,-q \
// RUN:   -Wl,--unresolved-symbols=ignore-all \
// RUN:   -fuse-ld=lld \
// RUN:   -nostdlib
// RUN: llvm-bolt %t.exe -o %t.bolt
// RUN: %clang %cflags -fPIC -pie %s -o %t_pie.exe -Wl,-q \
// RUN:   -Wl,--unresolved-symbols=ignore-all \
// RUN:   -target aarch64-linux -fuse-ld=lld \
// RUN:   -nostdlib
// RUN: llvm-bolt %t_pie.exe -o %t.bolt

// RUN: %clang %cflags -fPIC -shared %s -o %t.so -Wl,-q -fuse-ld=lld
// RUN: llvm-objdump -d -r --disassemble-symbols=main %t.so | FileCheck %s
// RUN: llvm-bolt %t.so -o %t.bolt.so

// Verify that unoptimized TLS access was generated for shared object.
// CHECK:      adrp    x0
// CHECK-NEXT: R_AARCH64_TLSDESC_ADR_PAGE21     tbssstruct
