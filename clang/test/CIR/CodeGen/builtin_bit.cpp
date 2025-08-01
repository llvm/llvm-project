// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int test_builtin_clrsb(int x) {
  return __builtin_clrsb(x);
}

// CIR-LABEL: _Z18test_builtin_clrsbi
// CIR:         [[TMP:%.+]] = cir.bit.clrsb(%{{.+}} : !s32i) : !s32i

// LLVM-LABEL: _Z18test_builtin_clrsbi
// LLVM:         %[[X:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[X_NEG:.+]] = icmp slt i32 %[[X]], 0
// LLVM-NEXT:    %[[X_NOT:.+]] = xor i32 %[[X]], -1
// LLVM-NEXT:    %[[P:.+]] = select i1 %[[X_NEG]], i32 %[[X_NOT]], i32 %[[X]]
// LLVM-NEXT:    %[[LZ:.+]] = call i32 @llvm.ctlz.i32(i32 %[[P]], i1 false)
// LLVM-NEXT:    %{{.+}} = sub i32 %[[LZ]], 1

// OGCG-LABEL: _Z18test_builtin_clrsbi
// OGCG:         %[[X:.+]] = load i32, ptr %{{.+}}, align 4
// OGCG-NEXT:    %[[X_NEG:.+]] = icmp slt i32 %[[X]], 0
// OGCG-NEXT:    %[[X_NOT:.+]] = xor i32 %[[X]], -1
// OGCG-NEXT:    %[[P:.+]] = select i1 %[[X_NEG]], i32 %[[X_NOT]], i32 %[[X]]
// OGCG-NEXT:    %[[LZ:.+]] = call i32 @llvm.ctlz.i32(i32 %[[P]], i1 false)
// OGCG-NEXT:    %{{.+}} = sub i32 %[[LZ]], 1

int test_builtin_clrsbl(long x) {
  return __builtin_clrsbl(x);
}

// CIR-LABEL: _Z19test_builtin_clrsbll
// CIR:         [[TMP:%.+]] = cir.bit.clrsb(%{{.+}} : !s64i) : !s64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !s64i), !s32i

// LLVM-LABEL: _Z19test_builtin_clrsbll
// LLVM:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[X_NEG:.+]] = icmp slt i64 %[[X]], 0
// LLVM-NEXT:    %[[X_NOT:.+]] = xor i64 %[[X]], -1
// LLVM-NEXT:    %[[P:.+]] = select i1 %[[X_NEG]], i64 %[[X_NOT]], i64 %[[X]]
// LLVM-NEXT:    %[[LZ:.+]] = call i64 @llvm.ctlz.i64(i64 %[[P]], i1 false)
// LLVM-NEXT:    %{{.+}} = sub i64 %[[LZ]], 1

// OGCG-LABEL: _Z19test_builtin_clrsbll
// OGCG:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[X_NEG:.+]] = icmp slt i64 %[[X]], 0
// OGCG-NEXT:    %[[X_NOT:.+]] = xor i64 %[[X]], -1
// OGCG-NEXT:    %[[P:.+]] = select i1 %[[X_NEG]], i64 %[[X_NOT]], i64 %[[X]]
// OGCG-NEXT:    %[[LZ:.+]] = call i64 @llvm.ctlz.i64(i64 %[[P]], i1 false)
// OGCG-NEXT:    %{{.+}} = sub i64 %[[LZ]], 1

int test_builtin_clrsbll(long long x) {
  return __builtin_clrsbll(x);
}

// CIR-LABEL: _Z20test_builtin_clrsbllx
// CIR:         [[TMP:%.+]] = cir.bit.clrsb(%{{.+}} : !s64i) : !s64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !s64i), !s32i

// LLVM-LABEL: _Z20test_builtin_clrsbllx
// LLVM:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[X_NEG:.+]] = icmp slt i64 %[[X]], 0
// LLVM-NEXT:    %[[X_NOT:.+]] = xor i64 %[[X]], -1
// LLVM-NEXT:    %[[P:.+]] = select i1 %[[X_NEG]], i64 %[[X_NOT]], i64 %[[X]]
// LLVM-NEXT:    %[[LZ:.+]] = call i64 @llvm.ctlz.i64(i64 %[[P]], i1 false)
// LLVM-NEXT:    %{{.+}} = sub i64 %[[LZ]], 1

// OGCG-LABEL: _Z20test_builtin_clrsbllx
// OGCG:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[X_NEG:.+]] = icmp slt i64 %[[X]], 0
// OGCG-NEXT:    %[[X_NOT:.+]] = xor i64 %[[X]], -1
// OGCG-NEXT:    %[[P:.+]] = select i1 %[[X_NEG]], i64 %[[X_NOT]], i64 %[[X]]
// OGCG-NEXT:    %[[LZ:.+]] = call i64 @llvm.ctlz.i64(i64 %[[P]], i1 false)
// OGCG-NEXT:    %{{.+}} = sub i64 %[[LZ]], 1

int test_builtin_ctzs(unsigned short x) {
  return __builtin_ctzs(x);
}

// CIR-LABEL: _Z17test_builtin_ctzst
// CIR:         [[TMP:%.+]] = cir.bit.ctz(%{{.+}} : !u16i) poison_zero : !u16i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u16i), !s32i

// LLVM-LABEL: _Z17test_builtin_ctzst
// LLVM:         %{{.+}} = call i16 @llvm.cttz.i16(i16 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_ctzst
// OGCG:         %{{.+}} = call i16 @llvm.cttz.i16(i16 %{{.+}}, i1 true)

int test_builtin_ctz(unsigned x) {
  return __builtin_ctz(x);
}

// CIR-LABEL: _Z16test_builtin_ctzj
// CIR:         [[TMP:%.+]] = cir.bit.ctz(%{{.+}} : !u32i) poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u32i), !s32i

// LLVM-LABEL: _Z16test_builtin_ctzj
// LLVM:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z16test_builtin_ctzj
// OGCG:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

int test_builtin_ctzl(unsigned long x) {
  return __builtin_ctzl(x);
}

// CIR-LABEL: _Z17test_builtin_ctzlm
// CIR:         [[TMP:%.+]] = cir.bit.ctz(%{{.+}} : !u64i) poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z17test_builtin_ctzlm
// LLVM:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_ctzlm
// OGCG:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

int test_builtin_ctzll(unsigned long long x) {
  return __builtin_ctzll(x);
}

// CIR-LABEL: _Z18test_builtin_ctzlly
// CIR:         [[TMP:%.+]] = cir.bit.ctz(%{{.+}} : !u64i) poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z18test_builtin_ctzlly
// LLVM:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z18test_builtin_ctzlly
// OGCG:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

int test_builtin_ctzg(unsigned x) {
  return __builtin_ctzg(x);
}

// CIR-LABEL: _Z17test_builtin_ctzgj
// CIR:         [[TMP:%.+]] = cir.bit.ctz(%{{.+}} : !u32i) poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u32i), !s32i

// LLVM-LABEL: _Z17test_builtin_ctzgj
// LLVM:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_ctzgj
// OGCG:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

int test_builtin_clzs(unsigned short x) {
  return __builtin_clzs(x);
}

// CIR-LABEL: _Z17test_builtin_clzst
// CIR:         [[TMP:%.+]] = cir.bit.clz(%{{.+}} : !u16i) poison_zero : !u16i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u16i), !s32i

// LLVM-LABEL: _Z17test_builtin_clzst
// LLVM:         %{{.+}} = call i16 @llvm.ctlz.i16(i16 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_clzst
// OGCG:         %{{.+}} = call i16 @llvm.ctlz.i16(i16 %{{.+}}, i1 true)

int test_builtin_clz(unsigned x) {
  return __builtin_clz(x);
}

// CIR-LABEL: _Z16test_builtin_clzj
// CIR:         [[TMP:%.+]] = cir.bit.clz(%{{.+}} : !u32i) poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u32i), !s32i

// LLVM-LABEL: _Z16test_builtin_clzj
// LLVM:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z16test_builtin_clzj
// OGCG:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

int test_builtin_clzl(unsigned long x) {
  return __builtin_clzl(x);
}

// CIR-LABEL: _Z17test_builtin_clzlm
// CIR:         [[TMP:%.+]] = cir.bit.clz(%{{.+}} : !u64i) poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z17test_builtin_clzlm
// LLVM:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_clzlm
// OGCG:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

int test_builtin_clzll(unsigned long long x) {
  return __builtin_clzll(x);
}

// CIR-LABEL: _Z18test_builtin_clzlly
// CIR:         [[TMP:%.+]] = cir.bit.clz(%{{.+}} : !u64i) poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z18test_builtin_clzlly
// LLVM:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z18test_builtin_clzlly
// OGCG:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

int test_builtin_clzg(unsigned x) {
  return __builtin_clzg(x);
}

// CIR-LABEL: _Z17test_builtin_clzgj
// CIR:         [[TMP:%.+]] = cir.bit.clz(%{{.+}} : !u32i) poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u32i), !s32i

// LLVM-LABEL: _Z17test_builtin_clzgj
// LLVM:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_clzgj
// OGCG:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

int test_builtin_parity(unsigned x) {
  return __builtin_parity(x);
}

// CIR-LABEL: _Z19test_builtin_parityj
// CIR:         [[TMP:%.+]] = cir.bit.parity(%{{.+}} : !u32i) : !u32i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u32i), !s32i

// LLVM-LABEL: _Z19test_builtin_parityj
// LLVM:         %[[X:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[POPCNT:.+]] = call i32 @llvm.ctpop.i32(i32 %[[X]])
// LLVM-NEXT:    %{{.+}} = and i32 %[[POPCNT]], 1

// OGCG-LABEL: _Z19test_builtin_parityj
// OGCG:         %[[X:.+]] = load i32, ptr %{{.+}}, align 4
// OGCG-NEXT:    %[[POPCNT:.+]] = call i32 @llvm.ctpop.i32(i32 %[[X]])
// OGCG-NEXT:    %{{.+}} = and i32 %[[POPCNT]], 1

int test_builtin_parityl(unsigned long x) {
  return __builtin_parityl(x);
}

// CIR-LABEL: _Z20test_builtin_paritylm
// CIR:         [[TMP:%.+]] = cir.bit.parity(%{{.+}} : !u64i) : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z20test_builtin_paritylm
// LLVM:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[POPCNT:.+]] = call i64 @llvm.ctpop.i64(i64 %[[X]])
// LLVM-NEXT:    %{{.+}} = and i64 %[[POPCNT]], 1

// OGCG-LABEL: _Z20test_builtin_paritylm
// OGCG:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[POPCNT:.+]] = call i64 @llvm.ctpop.i64(i64 %[[X]])
// OGCG-NEXT:    %{{.+}} = and i64 %[[POPCNT]], 1

int test_builtin_parityll(unsigned long long x) {
  return __builtin_parityll(x);
}

// CIR-LABEL: _Z21test_builtin_paritylly
// CIR:         [[TMP:%.+]] = cir.bit.parity(%{{.+}} : !u64i) : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z21test_builtin_paritylly
// LLVM:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[POPCNT:.+]] = call i64 @llvm.ctpop.i64(i64 %[[X]])
// LLVM-NEXT:    %{{.+}} = and i64 %[[POPCNT]], 1

// OGCG-LABEL: _Z21test_builtin_paritylly
// OGCG:         %[[X:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[POPCNT:.+]] = call i64 @llvm.ctpop.i64(i64 %[[X]])
// OGCG-NEXT:    %{{.+}} = and i64 %[[POPCNT]], 1

int test_builtin_popcount(unsigned x) {
  return __builtin_popcount(x);
}

// CIR-LABEL: _Z21test_builtin_popcountj
// CIR:         [[TMP:%.+]] = cir.bit.popcnt(%{{.+}} : !u32i) : !u32i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u32i), !s32i

// LLVM-LABEL: _Z21test_builtin_popcountj
// LLVM:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})

// OGCG-LABEL: _Z21test_builtin_popcountj
// OGCG:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})

int test_builtin_popcountl(unsigned long x) {
  return __builtin_popcountl(x);
}

// CIR-LABEL: _Z22test_builtin_popcountlm
// CIR:         [[TMP:%.+]] = cir.bit.popcnt(%{{.+}} : !u64i) : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z22test_builtin_popcountlm
// LLVM:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

// OGCG-LABEL: _Z22test_builtin_popcountlm
// OGCG:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

int test_builtin_popcountll(unsigned long long x) {
  return __builtin_popcountll(x);
}

// CIR-LABEL: _Z23test_builtin_popcountlly
// CIR:         [[TMP:%.+]] = cir.bit.popcnt(%{{.+}} : !u64i) : !u64i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u64i), !s32i

// LLVM-LABEL: _Z23test_builtin_popcountlly
// LLVM:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

// OGCG-LABEL: _Z23test_builtin_popcountlly
// OGCG:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

int test_builtin_popcountg(unsigned x) {
  return __builtin_popcountg(x);
}

// CIR-LABEL: _Z22test_builtin_popcountgj
// CIR:         [[TMP:%.+]] = cir.bit.popcnt(%{{.+}} : !u32i) : !u32i
// CIR:         {{%.+}} = cir.cast(integral, [[TMP]] : !u32i), !s32i

// LLVM-LABEL: _Z22test_builtin_popcountgj
// LLVM:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})

// OGCG-LABEL: _Z22test_builtin_popcountgj
// OGCG:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})
