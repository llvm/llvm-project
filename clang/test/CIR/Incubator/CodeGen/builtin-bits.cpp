// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck %s --check-prefix=CIR --input-file=%t.cir

int test_builtin_clrsb(int x) {
  return __builtin_clrsb(x);
}

// CIR-LABEL: _Z18test_builtin_clrsbi
// CIR: [[TMP:%.+]] = cir.clrsb %{{.+}} : !s32i

int test_builtin_clrsbl(long x) {
  return __builtin_clrsbl(x);
}

// CIR-LABEL: _Z19test_builtin_clrsbll
// CIR: [[TMP:%.+]] = cir.clrsb %{{.+}} : !s64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !s64i -> !s32i

int test_builtin_clrsbll(long long x) {
  return __builtin_clrsbll(x);
}

// CIR-LABEL: _Z20test_builtin_clrsbllx
// CIR: [[TMP:%.+]] = cir.clrsb %{{.+}} : !s64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !s64i -> !s32i

int test_builtin_ctzs(unsigned short x) {
  return __builtin_ctzs(x);
}

// CIR-LABEL: _Z17test_builtin_ctzst
// CIR: [[TMP:%.+]] = cir.ctz %{{.+}} zero_poison : !u16i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u16i -> !s32i

int test_builtin_ctz(unsigned x) {
  return __builtin_ctz(x);
}

// CIR-LABEL: _Z16test_builtin_ctzj
// CIR: [[TMP:%.+]] = cir.ctz %{{.+}} zero_poison : !u32i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

int test_builtin_ctzl(unsigned long x) {
  return __builtin_ctzl(x);
}

// CIR-LABEL: _Z17test_builtin_ctzlm
// CIR: [[TMP:%.+]] = cir.ctz %{{.+}} zero_poison : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_ctzll(unsigned long long x) {
  return __builtin_ctzll(x);
}

// CIR-LABEL: _Z18test_builtin_ctzlly
// CIR: [[TMP:%.+]] = cir.ctz %{{.+}} zero_poison : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_ctzg(unsigned x) {
  return __builtin_ctzg(x);
}

// CIR-LABEL: _Z17test_builtin_ctzgj
// CIR: [[TMP:%.+]] = cir.ctz %{{.+}} zero_poison : !u32i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

int test_builtin_clzs(unsigned short x) {
  return __builtin_clzs(x);
}

// CIR-LABEL: _Z17test_builtin_clzst
// CIR: [[TMP:%.+]] = cir.clz %{{.+}} zero_poison : !u16i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u16i -> !s32i

int test_builtin_clz(unsigned x) {
  return __builtin_clz(x);
}

// CIR-LABEL: _Z16test_builtin_clzj
// CIR: [[TMP:%.+]] = cir.clz %{{.+}} zero_poison : !u32i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

int test_builtin_clzl(unsigned long x) {
  return __builtin_clzl(x);
}

// CIR-LABEL: _Z17test_builtin_clzlm
// CIR: [[TMP:%.+]] = cir.clz %{{.+}} zero_poison : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_clzll(unsigned long long x) {
  return __builtin_clzll(x);
}

// CIR-LABEL: _Z18test_builtin_clzlly
// CIR: [[TMP:%.+]] = cir.clz %{{.+}} zero_poison : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_clzg(unsigned x) {
  return __builtin_clzg(x);
}

// CIR-LABEL: _Z17test_builtin_clzgj
// CIR: [[TMP:%.+]] = cir.clz %{{.+}} zero_poison : !u32i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

int test_builtin_ffs(int x) {
  return __builtin_ffs(x);
}

// CIR-LABEL: _Z16test_builtin_ffsi
// CIR: [[TMP:%.+]] = cir.ffs %{{.+}} : !s32i

int test_builtin_ffsl(long x) {
  return __builtin_ffsl(x);
}

// CIR-LABEL: _Z17test_builtin_ffsll
// CIR: [[TMP:%.+]] = cir.ffs %{{.+}} : !s64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !s64i -> !s32i

int test_builtin_ffsll(long long x) {
  return __builtin_ffsll(x);
}

// CIR-LABEL: _Z18test_builtin_ffsllx
// CIR: [[TMP:%.+]] = cir.ffs %{{.+}} : !s64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !s64i -> !s32i

int test_builtin_parity(unsigned x) {
  return __builtin_parity(x);
}

// CIR-LABEL: _Z19test_builtin_parityj
// CIR: [[TMP:%.+]] = cir.parity %{{.+}} : !u32i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

int test_builtin_parityl(unsigned long x) {
  return __builtin_parityl(x);
}

// CIR-LABEL: _Z20test_builtin_paritylm
// CIR: [[TMP:%.+]] = cir.parity %{{.+}} : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_parityll(unsigned long long x) {
  return __builtin_parityll(x);
}

// CIR-LABEL: _Z21test_builtin_paritylly
// CIR: [[TMP:%.+]] = cir.parity %{{.+}} : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_popcount(unsigned x) {
  return __builtin_popcount(x);
}

// CIR-LABEL: _Z21test_builtin_popcountj
// CIR: [[TMP:%.+]] = cir.popcount %{{.+}} : !u32i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

int test_builtin_popcountl(unsigned long x) {
  return __builtin_popcountl(x);
}

// CIR-LABEL: _Z22test_builtin_popcountlm
// CIR: [[TMP:%.+]] = cir.popcount %{{.+}} : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_popcountll(unsigned long long x) {
  return __builtin_popcountll(x);
}

// CIR-LABEL: _Z23test_builtin_popcountlly
// CIR: [[TMP:%.+]] = cir.popcount %{{.+}} : !u64i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

int test_builtin_popcountg(unsigned x) {
  return __builtin_popcountg(x);
}

// CIR-LABEL: _Z22test_builtin_popcountgj
// CIR: [[TMP:%.+]] = cir.popcount %{{.+}} : !u32i
// CIR: {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i
