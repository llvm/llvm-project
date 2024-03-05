// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++17 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

int test_builtin_clrsb(int x) {
  return __builtin_clrsb(x);
}

// CHECK: cir.func @_Z18test_builtin_clrsbi
// CHECK:   %{{.+}} = cir.bit.clrsb(%{{.+}} : !s32i) : !s32i
// CHECK: }

int test_builtin_clrsbl(long x) {
  return __builtin_clrsbl(x);
}

// CHECK: cir.func @_Z19test_builtin_clrsbll
// CHECK:   %{{.+}} = cir.bit.clrsb(%{{.+}} : !s64i) : !s32i
// CHECK: }

int test_builtin_clrsbll(long long x) {
  return __builtin_clrsbll(x);
}

// CHECK: cir.func @_Z20test_builtin_clrsbllx
// CHECK:   %{{.+}} = cir.bit.clrsb(%{{.+}} : !s64i) : !s32i
// CHECK: }

int test_builtin_ctzs(unsigned short x) {
  return __builtin_ctzs(x);
}

// CHECK: cir.func @_Z17test_builtin_ctzst
// CHECK:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u16i) : !s32i
// CHEKC: }

int test_builtin_ctz(unsigned x) {
  return __builtin_ctz(x);
}

// CHECK: cir.func @_Z16test_builtin_ctzj
// CHECK:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u32i) : !s32i
// CHECK: }

int test_builtin_ctzl(unsigned long x) {
  return __builtin_ctzl(x);
}

// CHECK: cir.func @_Z17test_builtin_ctzlm
// CHECK:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u64i) : !s32i
// CHECK: }

int test_builtin_ctzll(unsigned long long x) {
  return __builtin_ctzll(x);
}

// CHECK: cir.func @_Z18test_builtin_ctzlly
// CHECK:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u64i) : !s32i
// CHECK: }

int test_builtin_clzs(unsigned short x) {
  return __builtin_clzs(x);
}

// CHECK: cir.func @_Z17test_builtin_clzst
// CHECK:   %{{.+}} = cir.bit.clz(%{{.+}} : !u16i) : !s32i
// CHECK: }

int test_builtin_clz(unsigned x) {
  return __builtin_clz(x);
}

// CHECK: cir.func @_Z16test_builtin_clzj
// CHECK:   %{{.+}} = cir.bit.clz(%{{.+}} : !u32i) : !s32i
// CHECK: }

int test_builtin_clzl(unsigned long x) {
  return __builtin_clzl(x);
}

// CHECK: cir.func @_Z17test_builtin_clzlm
// CHECK:   %{{.+}} = cir.bit.clz(%{{.+}} : !u64i) : !s32i
// CHECK: }

int test_builtin_clzll(unsigned long long x) {
  return __builtin_clzll(x);
}

// CHECK: cir.func @_Z18test_builtin_clzlly
// CHECK:   %{{.+}} = cir.bit.clz(%{{.+}} : !u64i) : !s32i
// CHECK: }

int test_builtin_ffs(int x) {
  return __builtin_ffs(x);
}

// CHECK: cir.func @_Z16test_builtin_ffsi
// CHECK:   %{{.+}} = cir.bit.ffs(%{{.+}} : !s32i) : !s32i
// CHECK: }

int test_builtin_ffsl(long x) {
  return __builtin_ffsl(x);
}

// CHECK: cir.func @_Z17test_builtin_ffsll
// CHECK:   %{{.+}} = cir.bit.ffs(%{{.+}} : !s64i) : !s32i
// CHECK: }

int test_builtin_ffsll(long long x) {
  return __builtin_ffsll(x);
}

// CHECK: cir.func @_Z18test_builtin_ffsllx
// CHECK:   %{{.+}} = cir.bit.ffs(%{{.+}} : !s64i) : !s32i
// CHECK: }

int test_builtin_parity(unsigned x) {
  return __builtin_parity(x);
}

// CHECK: cir.func @_Z19test_builtin_parityj
// CHECK:   %{{.+}} = cir.bit.parity(%{{.+}} : !u32i) : !s32i
// CHECK: }

int test_builtin_parityl(unsigned long x) {
  return __builtin_parityl(x);
}

// CHECK: cir.func @_Z20test_builtin_paritylm
// CHECK:   %{{.+}} = cir.bit.parity(%{{.+}} : !u64i) : !s32i
// CHECK: }

int test_builtin_parityll(unsigned long long x) {
  return __builtin_parityll(x);
}

// CHECK: cir.func @_Z21test_builtin_paritylly
// CHECK:   %{{.+}} = cir.bit.parity(%{{.+}} : !u64i) : !s32i
// CHECK: }

int test_builtin_popcount(unsigned x) {
  return __builtin_popcount(x);
}

// CHECK: cir.func @_Z21test_builtin_popcountj
// CHECK:   %{{.+}} = cir.bit.popcount(%{{.+}} : !u32i) : !s32i
// CHECK: }

int test_builtin_popcountl(unsigned long x) {
  return __builtin_popcountl(x);
}

// CHECK: cir.func @_Z22test_builtin_popcountlm
// CHECK:   %{{.+}} = cir.bit.popcount(%{{.+}} : !u64i) : !s32i
// CHECK: }

int test_builtin_popcountll(unsigned long long x) {
  return __builtin_popcountll(x);
}

// CHECK: cir.func @_Z23test_builtin_popcountlly
// CHECK:   %{{.+}} = cir.bit.popcount(%{{.+}} : !u64i) : !s32i
// CHECK: }
