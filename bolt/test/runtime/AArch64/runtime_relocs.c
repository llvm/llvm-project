// This test checks dynamic relocations support for aarch64.

// RUN: %clang %cflags -pie -fPIC %S/Inputs/runtime_relocs.c \
// RUN:    -shared -fuse-ld=lld -o %t.so -Wl,-q -Wl,-soname=rel.so
// RUN: %clang %cflags -no-pie %s -fuse-ld=lld \
// RUN:    -o %t.exe -Wl,-q %t.so
// RUN: llvm-bolt %t.so -o %t.bolt.so -use-old-text=0 -lite=0
// RUN: llvm-bolt %t.exe -o %t.bolt.exe -use-old-text=0 -lite=0
// RUN: LD_PRELOAD=%t.bolt.so %t.bolt.exe

// Check relocations in library:
//
// RUN: llvm-readelf -r %t.bolt.so | FileCheck %s -check-prefix=CHECKLIB
//
// CHECKLIB: 0000000600000401 R_AARCH64_GLOB_DAT     {{.*}} a + 0
// CHECKLIB: 0000000700000407 R_AARCH64_TLSDESC      {{.*}} t1 + 0
// CHECKLIB: 0000000600000101 R_AARCH64_ABS64        {{.*}} a + 0

// Check relocations in executable:
//
// RUN: llvm-readelf -r %t.bolt.exe | FileCheck %s -check-prefix=CHECKEXE
//
// CHECKEXE: 0000000600000406 R_AARCH64_TLS_TPREL64  {{.*}} t1 + 0
// CHECKEXE: 0000000800000400 R_AARCH64_COPY         {{.*}} a + 0
// CHECKEXE: 0000000700000402 R_AARCH64_JUMP_SLOT    {{.*}} inc + 0

// Check traditional TLS relocations R_AARCH64_TLS_DTPMOD64 and
// R_AARCH64_TLS_DTPREL64 emitted correctly after bolt. Since these
// relocations are obsolete and clang and lld does not support them,
// the initial binary was built with gcc and ld with -mtls-dialect=trad flag.
//
// RUN: yaml2obj %p/Inputs/tls_trad.yaml &> %t.trad.so
// RUN: llvm-bolt %t.trad.so -o %t.trad.bolt.so -use-old-text=0 -lite=0
// RUN: llvm-readelf -r %t.trad.so | FileCheck %s -check-prefix=CHECKTRAD
//
// CHECKTRAD: 0000000100000404 R_AARCH64_TLS_DTPMOD64 {{.*}} t1 + 0
// CHECKTRAD: 0000000100000405 R_AARCH64_TLS_DTPREL64 {{.*}} t1 + 0

// The ld linker emits R_AARCH64_TLSDESC to .rela.plt section, check that
// it is emitted correctly.
//
// RUN: yaml2obj %p/Inputs/tls_ld.yaml &> %t.ld.so
// RUN: llvm-bolt %t.ld.so -o %t.ld.bolt.so -use-old-text=0 -lite=0
// RUN: llvm-readelf -r %t.ld.bolt.so | FileCheck %s -check-prefix=CHECKLD
//
// CHECKLD: 0000000100000407 R_AARCH64_TLSDESC        {{.*}} t1 + 0

extern int a; // R_*_COPY

extern __thread int t1; // R_*_TLS_TPREL64

int inc(int a); // R_*_JUMP_SLOT

int dec(int a) { return a - 1; }

void *resolver() { return dec; }

int ifuncDec(int a) __attribute__((ifunc("resolver"))); // R_*_IRELATIVE

int main() {
  ++t1;
  ifuncDec(a);
  inc(a);
}
