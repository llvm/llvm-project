// REQUIRES: hexagon-registered-target
// RUN: %clang_cc1 -triple hexagon-unknown-elf -emit-llvm %s -o - | FileCheck %s

// CHECK-LABEL: test1
// CHECK: @llvm.hexagon.L2.loadrub.pci
unsigned char test1(int mod, void *start) {
  unsigned char *base = start;
  return __builtin_HEXAGON_L2_loadrub_pci(&base, 4, mod, start);
}

// CHECK-LABEL: test2
// CHECK: @llvm.hexagon.L2.loadrb.pci
unsigned char test2(int mod, void *start) {
  char *base = start;
  return __builtin_HEXAGON_L2_loadrb_pci(&base, 4, mod, start);
}

// CHECK-LABEL: test3
// CHECK: @llvm.hexagon.L2.loadruh.pci
unsigned short test3(int mod, void *start) {
  unsigned short *base = start;
  return __builtin_HEXAGON_L2_loadruh_pci(&base, 4, mod, start);
}

// CHECK-LABEL: test4
// CHECK: @llvm.hexagon.L2.loadrh.pci
short test4(int mod, void *start) {
  short *base = start;
  return __builtin_HEXAGON_L2_loadrh_pci(&base, 4, mod, start);
}

// CHECK-LABEL: test5
// CHECK: @llvm.hexagon.L2.loadri.pci
int test5(int mod, void *start) {
  int *base = start;
  return __builtin_HEXAGON_L2_loadri_pci(&base, 4, mod, start);
}

// CHECK-LABEL: test6
// CHECK: @llvm.hexagon.L2.loadrd.pci
long long test6(int mod, void *start) {
  long long *base = start;
  return __builtin_HEXAGON_L2_loadrd_pci(&base, 8, mod, start);
}

// CHECK-LABEL: test7
// CHECK: @llvm.hexagon.L2.loadrub.pcr
unsigned char test7(int mod, void *start) {
  unsigned char *base = start;
  return __builtin_HEXAGON_L2_loadrub_pcr(&base, mod, start);
}

// CHECK-LABEL: test8
// CHECK: @llvm.hexagon.L2.loadrb.pcr
unsigned char test8(int mod, void *start) {
  char *base = start;
  return __builtin_HEXAGON_L2_loadrb_pcr(&base, mod, start);
}

// CHECK-LABEL: test9
// CHECK: @llvm.hexagon.L2.loadruh.pcr
unsigned short test9(int mod, void *start) {
  unsigned short *base = start;
  return __builtin_HEXAGON_L2_loadruh_pcr(&base, mod, start);
}

// CHECK-LABEL: test10
// CHECK: @llvm.hexagon.L2.loadrh.pcr
short test10(int mod, void *start) {
  short *base = start;
  return __builtin_HEXAGON_L2_loadrh_pcr(&base, mod, start);
}

// CHECK-LABEL: test11
// CHECK: @llvm.hexagon.L2.loadri.pcr
int test11(int mod, void *start) {
  int *base = start;
  return __builtin_HEXAGON_L2_loadri_pcr(&base, mod, start);
}

// CHECK-LABEL: test12
// CHECK: @llvm.hexagon.L2.loadrd.pcr
long long test12(int mod, void *start) {
  long long *base = start;
  return __builtin_HEXAGON_L2_loadrd_pcr(&base, mod, start);
}

// CHECK-LABEL: test13
// CHECK: @llvm.hexagon.S2.storerb.pci
void test13(int mod, void *start, char v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerb_pci(&base, 4, mod, v, start);
}

// CHECK-LABEL: test14
// CHECK: @llvm.hexagon.S2.storerh.pci
void test14(int mod, void *start, short v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerh_pci(&base, 4, mod, v, start);
}

// CHECK-LABEL: test15
// CHECK: @llvm.hexagon.S2.storerf.pci
void test15(int mod, void *start, short v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerf_pci(&base, 4, mod, v, start);
}

// CHECK-LABEL: test16
// CHECK: @llvm.hexagon.S2.storeri.pci
void test16(int mod, void *start, int v) {
  void *base = start;
  __builtin_HEXAGON_S2_storeri_pci(&base, 4, mod, v, start);
}

// CHECK-LABEL: test17
// CHECK: @llvm.hexagon.S2.storerd.pci
void test17(int mod, void *start, long long v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerd_pci(&base, 8, mod, v, start);
}

// CHECK-LABEL: test18
// CHECK: @llvm.hexagon.S2.storerb.pcr
void test18(int mod, void *start, char v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerb_pcr(&base, mod, v, start);
}

// CHECK-LABEL: test19
// CHECK: @llvm.hexagon.S2.storerh.pcr
void test19(int mod, void *start, short v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerh_pcr(&base, mod, v, start);
}

// CHECK-LABEL: test20
// CHECK: @llvm.hexagon.S2.storerf.pcr
void test20(int mod, void *start, short v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerf_pcr(&base, mod, v, start);
}

// CHECK-LABEL: test21
// CHECK: @llvm.hexagon.S2.storeri.pcr
void test21(int mod, void *start, int v) {
  void *base = start;
  __builtin_HEXAGON_S2_storeri_pcr(&base, mod, v, start);
}

// CHECK-LABEL: test22
// CHECK: @llvm.hexagon.S2.storerd.pcr
void test22(int mod, void *start, long long v) {
  void *base = start;
  __builtin_HEXAGON_S2_storerd_pcr(&base, mod, v, start);
}
