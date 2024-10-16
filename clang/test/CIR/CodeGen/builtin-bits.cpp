// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir

int test_builtin_clrsb(int x) {
  return __builtin_clrsb(x);
}

// CIR: cir.func @_Z18test_builtin_clrsbi
// CIR:   %{{.+}} = cir.bit.clrsb(%{{.+}} : !s32i) : !s32i
// CIR: }

int test_builtin_clrsbl(long x) {
  return __builtin_clrsbl(x);
}

// CIR: cir.func @_Z19test_builtin_clrsbll
// CIR:   %{{.+}} = cir.bit.clrsb(%{{.+}} : !s64i) : !s32i
// CIR: }

int test_builtin_clrsbll(long long x) {
  return __builtin_clrsbll(x);
}

// CIR: cir.func @_Z20test_builtin_clrsbllx
// CIR:   %{{.+}} = cir.bit.clrsb(%{{.+}} : !s64i) : !s32i
// CIR: }

int test_builtin_ctzs(unsigned short x) {
  return __builtin_ctzs(x);
}

// CIR: cir.func @_Z17test_builtin_ctzst
// CIR:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u16i) : !s32i
// CHEKC: }

int test_builtin_ctz(unsigned x) {
  return __builtin_ctz(x);
}

// CIR: cir.func @_Z16test_builtin_ctzj
// CIR:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u32i) : !s32i
// CIR: }

int test_builtin_ctzl(unsigned long x) {
  return __builtin_ctzl(x);
}

// CIR: cir.func @_Z17test_builtin_ctzlm
// CIR:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_ctzll(unsigned long long x) {
  return __builtin_ctzll(x);
}

// CIR: cir.func @_Z18test_builtin_ctzlly
// CIR:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_ctzg(unsigned x) {
  return __builtin_ctzg(x);
}

// CIR: cir.func @_Z17test_builtin_ctzgj
// CIR:   %{{.+}} = cir.bit.ctz(%{{.+}} : !u32i) : !s32i
// CIR: }

int test_builtin_clzs(unsigned short x) {
  return __builtin_clzs(x);
}

// CIR: cir.func @_Z17test_builtin_clzst
// CIR:   %{{.+}} = cir.bit.clz(%{{.+}} : !u16i) : !s32i
// CIR: }

int test_builtin_clz(unsigned x) {
  return __builtin_clz(x);
}

// CIR: cir.func @_Z16test_builtin_clzj
// CIR:   %{{.+}} = cir.bit.clz(%{{.+}} : !u32i) : !s32i
// CIR: }

int test_builtin_clzl(unsigned long x) {
  return __builtin_clzl(x);
}

// CIR: cir.func @_Z17test_builtin_clzlm
// CIR:   %{{.+}} = cir.bit.clz(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_clzll(unsigned long long x) {
  return __builtin_clzll(x);
}

// CIR: cir.func @_Z18test_builtin_clzlly
// CIR:   %{{.+}} = cir.bit.clz(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_clzg(unsigned x) {
  return __builtin_clzg(x);
}

// CIR: cir.func @_Z17test_builtin_clzgj
// CIR:   %{{.+}} = cir.bit.clz(%{{.+}} : !u32i) : !s32i
// CIR: }

int test_builtin_ffs(int x) {
  return __builtin_ffs(x);
}

// CIR: cir.func @_Z16test_builtin_ffsi
// CIR:   %{{.+}} = cir.bit.ffs(%{{.+}} : !s32i) : !s32i
// CIR: }

int test_builtin_ffsl(long x) {
  return __builtin_ffsl(x);
}

// CIR: cir.func @_Z17test_builtin_ffsll
// CIR:   %{{.+}} = cir.bit.ffs(%{{.+}} : !s64i) : !s32i
// CIR: }

int test_builtin_ffsll(long long x) {
  return __builtin_ffsll(x);
}

// CIR: cir.func @_Z18test_builtin_ffsllx
// CIR:   %{{.+}} = cir.bit.ffs(%{{.+}} : !s64i) : !s32i
// CIR: }

int test_builtin_parity(unsigned x) {
  return __builtin_parity(x);
}

// CIR: cir.func @_Z19test_builtin_parityj
// CIR:   %{{.+}} = cir.bit.parity(%{{.+}} : !u32i) : !s32i
// CIR: }

int test_builtin_parityl(unsigned long x) {
  return __builtin_parityl(x);
}

// CIR: cir.func @_Z20test_builtin_paritylm
// CIR:   %{{.+}} = cir.bit.parity(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_parityll(unsigned long long x) {
  return __builtin_parityll(x);
}

// CIR: cir.func @_Z21test_builtin_paritylly
// CIR:   %{{.+}} = cir.bit.parity(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_popcount(unsigned x) {
  return __builtin_popcount(x);
}

// CIR: cir.func @_Z21test_builtin_popcountj
// CIR:   %{{.+}} = cir.bit.popcount(%{{.+}} : !u32i) : !s32i
// CIR: }

int test_builtin_popcountl(unsigned long x) {
  return __builtin_popcountl(x);
}

// CIR: cir.func @_Z22test_builtin_popcountlm
// CIR:   %{{.+}} = cir.bit.popcount(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_popcountll(unsigned long long x) {
  return __builtin_popcountll(x);
}

// CIR: cir.func @_Z23test_builtin_popcountlly
// CIR:   %{{.+}} = cir.bit.popcount(%{{.+}} : !u64i) : !s32i
// CIR: }

int test_builtin_popcountg(unsigned x) {
  return __builtin_popcountg(x);
}

// CIR: cir.func @_Z22test_builtin_popcountgj
// CIR:   %{{.+}} = cir.bit.popcount(%{{.+}} : !u32i) : !s32i
// CIR: }
