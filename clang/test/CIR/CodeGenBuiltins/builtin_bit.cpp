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
// CIR:         [[TMP:%.+]] = cir.clrsb %{{.+}} : !s32i

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
// CIR:         [[TMP:%.+]] = cir.clrsb %{{.+}} : !s64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !s64i -> !s32i

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
// CIR:         [[TMP:%.+]] = cir.clrsb %{{.+}} : !s64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !s64i -> !s32i

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
// CIR:         [[TMP:%.+]] = cir.ctz %{{.+}} poison_zero : !u16i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u16i -> !s32i

// LLVM-LABEL: _Z17test_builtin_ctzst
// LLVM:         %{{.+}} = call i16 @llvm.cttz.i16(i16 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_ctzst
// OGCG:         %{{.+}} = call i16 @llvm.cttz.i16(i16 %{{.+}}, i1 true)

int test_builtin_ctz(unsigned x) {
  return __builtin_ctz(x);
}

// CIR-LABEL: _Z16test_builtin_ctzj
// CIR:         [[TMP:%.+]] = cir.ctz %{{.+}} poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

// LLVM-LABEL: _Z16test_builtin_ctzj
// LLVM:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z16test_builtin_ctzj
// OGCG:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

int test_builtin_ctzl(unsigned long x) {
  return __builtin_ctzl(x);
}

// CIR-LABEL: _Z17test_builtin_ctzlm
// CIR:         [[TMP:%.+]] = cir.ctz %{{.+}} poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

// LLVM-LABEL: _Z17test_builtin_ctzlm
// LLVM:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_ctzlm
// OGCG:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

int test_builtin_ctzll(unsigned long long x) {
  return __builtin_ctzll(x);
}

// CIR-LABEL: _Z18test_builtin_ctzlly
// CIR:         [[TMP:%.+]] = cir.ctz %{{.+}} poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

// LLVM-LABEL: _Z18test_builtin_ctzlly
// LLVM:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z18test_builtin_ctzlly
// OGCG:         %{{.+}} = call i64 @llvm.cttz.i64(i64 %{{.+}}, i1 true)

int test_builtin_ctzg(unsigned x) {
  return __builtin_ctzg(x);
}

// CIR-LABEL: _Z17test_builtin_ctzgj
// CIR:         [[TMP:%.+]] = cir.ctz %{{.+}} poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

// LLVM-LABEL: _Z17test_builtin_ctzgj
// LLVM:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_ctzgj
// OGCG:         %{{.+}} = call i32 @llvm.cttz.i32(i32 %{{.+}}, i1 true)

int test_builtin_clzs(unsigned short x) {
  return __builtin_clzs(x);
}

// CIR-LABEL: _Z17test_builtin_clzst
// CIR:         [[TMP:%.+]] = cir.clz %{{.+}} poison_zero : !u16i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u16i -> !s32i

// LLVM-LABEL: _Z17test_builtin_clzst
// LLVM:         %{{.+}} = call i16 @llvm.ctlz.i16(i16 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_clzst
// OGCG:         %{{.+}} = call i16 @llvm.ctlz.i16(i16 %{{.+}}, i1 true)

int test_builtin_clz(unsigned x) {
  return __builtin_clz(x);
}

// CIR-LABEL: _Z16test_builtin_clzj
// CIR:         [[TMP:%.+]] = cir.clz %{{.+}} poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

// LLVM-LABEL: _Z16test_builtin_clzj
// LLVM:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z16test_builtin_clzj
// OGCG:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

int test_builtin_clzl(unsigned long x) {
  return __builtin_clzl(x);
}

// CIR-LABEL: _Z17test_builtin_clzlm
// CIR:         [[TMP:%.+]] = cir.clz %{{.+}} poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

// LLVM-LABEL: _Z17test_builtin_clzlm
// LLVM:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_clzlm
// OGCG:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

int test_builtin_clzll(unsigned long long x) {
  return __builtin_clzll(x);
}

// CIR-LABEL: _Z18test_builtin_clzlly
// CIR:         [[TMP:%.+]] = cir.clz %{{.+}} poison_zero : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

// LLVM-LABEL: _Z18test_builtin_clzlly
// LLVM:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

// OGCG-LABEL: _Z18test_builtin_clzlly
// OGCG:         %{{.+}} = call i64 @llvm.ctlz.i64(i64 %{{.+}}, i1 true)

int test_builtin_clzg(unsigned x) {
  return __builtin_clzg(x);
}

// CIR-LABEL: _Z17test_builtin_clzgj
// CIR:         [[TMP:%.+]] = cir.clz %{{.+}} poison_zero : !u32i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

// LLVM-LABEL: _Z17test_builtin_clzgj
// LLVM:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

// OGCG-LABEL: _Z17test_builtin_clzgj
// OGCG:         %{{.+}} = call i32 @llvm.ctlz.i32(i32 %{{.+}}, i1 true)

int test_builtin_ffs(int x) {
  return __builtin_ffs(x);
}

// CIR-LABEL: _Z16test_builtin_ffsi
// CIR:         %{{.+}} = cir.ffs %{{.+}} : !s32i
// CIR:       }

// LLVM-LABEL: _Z16test_builtin_ffsi
// LLVM:         %[[INPUT:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[CTZ:.+]] = call i32 @llvm.cttz.i32(i32 %[[INPUT]], i1 true)
// LLVM-NEXT:    %[[R1:.+]] = add i32 %[[CTZ]], 1
// LLVM-NEXT:    %[[IS_ZERO:.+]] = icmp eq i32 %[[INPUT]], 0
// LLVM-NEXT:    %{{.+}} = select i1 %[[IS_ZERO]], i32 0, i32 %[[R1]]
// LLVM:       }

// OGCG-LABEL: _Z16test_builtin_ffsi
// OGCG:         %[[INPUT:.+]] = load i32, ptr %{{.+}}, align 4
// OGCG-NEXT:    %[[CTZ:.+]] = call i32 @llvm.cttz.i32(i32 %[[INPUT]], i1 true)
// OGCG-NEXT:    %[[R1:.+]] = add i32 %[[CTZ]], 1
// OGCG-NEXT:    %[[IS_ZERO:.+]] = icmp eq i32 %[[INPUT]], 0
// OGCG-NEXT:    %{{.+}} = select i1 %[[IS_ZERO]], i32 0, i32 %[[R1]]
// OGCG:       }

int test_builtin_ffsl(long x) {
  return __builtin_ffsl(x);
}

// CIR-LABEL: _Z17test_builtin_ffsll
// CIR:         %{{.+}} = cir.ffs %{{.+}} : !s64i
// CIR:       }

// LLVM-LABEL: _Z17test_builtin_ffsll
// LLVM:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[CTZ:.+]] = call i64 @llvm.cttz.i64(i64 %[[INPUT]], i1 true)
// LLVM-NEXT:    %[[R1:.+]] = add i64 %[[CTZ]], 1
// LLVM-NEXT:    %[[IS_ZERO:.+]] = icmp eq i64 %[[INPUT]], 0
// LLVM-NEXT:    %{{.+}} = select i1 %[[IS_ZERO]], i64 0, i64 %[[R1]]
// LLVM:       }

// OGCG-LABEL: _Z17test_builtin_ffsll
// OGCG:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[CTZ:.+]] = call i64 @llvm.cttz.i64(i64 %[[INPUT]], i1 true)
// OGCG-NEXT:    %[[R1:.+]] = add i64 %[[CTZ]], 1
// OGCG-NEXT:    %[[IS_ZERO:.+]] = icmp eq i64 %[[INPUT]], 0
// OGCG-NEXT:    %{{.+}} = select i1 %[[IS_ZERO]], i64 0, i64 %[[R1]]
// OGCG:       }

int test_builtin_ffsll(long long x) {
  return __builtin_ffsll(x);
}

// CIR-LABEL: _Z18test_builtin_ffsllx
// CIR:         %{{.+}} = cir.ffs %{{.+}} : !s64i
// CIR:       }

// LLVM-LABEL: _Z18test_builtin_ffsllx
// LLVM:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[CTZ:.+]] = call i64 @llvm.cttz.i64(i64 %[[INPUT]], i1 true)
// LLVM-NEXT:    %[[R1:.+]] = add i64 %[[CTZ]], 1
// LLVM-NEXT:    %[[IS_ZERO:.+]] = icmp eq i64 %[[INPUT]], 0
// LLVM-NEXT:    %{{.+}} = select i1 %[[IS_ZERO]], i64 0, i64 %[[R1]]
// LLVM:       }

// OGCG-LABEL: _Z18test_builtin_ffsllx
// OGCG:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[CTZ:.+]] = call i64 @llvm.cttz.i64(i64 %[[INPUT]], i1 true)
// OGCG-NEXT:    %[[R1:.+]] = add i64 %[[CTZ]], 1
// OGCG-NEXT:    %[[IS_ZERO:.+]] = icmp eq i64 %[[INPUT]], 0
// OGCG-NEXT:    %{{.+}} = select i1 %[[IS_ZERO]], i64 0, i64 %[[R1]]
// OGCG:       }

int test_builtin_parity(unsigned x) {
  return __builtin_parity(x);
}

// CIR-LABEL: _Z19test_builtin_parityj
// CIR:         [[TMP:%.+]] = cir.parity %{{.+}} : !u32i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

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
// CIR:         [[TMP:%.+]] = cir.parity %{{.+}} : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

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
// CIR:         [[TMP:%.+]] = cir.parity %{{.+}} : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

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
// CIR:         [[TMP:%.+]] = cir.popcount %{{.+}} : !u32i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

// LLVM-LABEL: _Z21test_builtin_popcountj
// LLVM:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})

// OGCG-LABEL: _Z21test_builtin_popcountj
// OGCG:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})

int test_builtin_popcountl(unsigned long x) {
  return __builtin_popcountl(x);
}

// CIR-LABEL: _Z22test_builtin_popcountlm
// CIR:         [[TMP:%.+]] = cir.popcount %{{.+}} : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

// LLVM-LABEL: _Z22test_builtin_popcountlm
// LLVM:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

// OGCG-LABEL: _Z22test_builtin_popcountlm
// OGCG:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

int test_builtin_popcountll(unsigned long long x) {
  return __builtin_popcountll(x);
}

// CIR-LABEL: _Z23test_builtin_popcountlly
// CIR:         [[TMP:%.+]] = cir.popcount %{{.+}} : !u64i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u64i -> !s32i

// LLVM-LABEL: _Z23test_builtin_popcountlly
// LLVM:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

// OGCG-LABEL: _Z23test_builtin_popcountlly
// OGCG:         %{{.+}} = call i64 @llvm.ctpop.i64(i64 %{{.+}})

int test_builtin_popcountg(unsigned x) {
  return __builtin_popcountg(x);
}

// CIR-LABEL: _Z22test_builtin_popcountgj
// CIR:         [[TMP:%.+]] = cir.popcount %{{.+}} : !u32i
// CIR:         {{%.+}} = cir.cast integral [[TMP]] : !u32i -> !s32i

// LLVM-LABEL: _Z22test_builtin_popcountgj
// LLVM:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})

// OGCG-LABEL: _Z22test_builtin_popcountgj
// OGCG:         %{{.+}} = call i32 @llvm.ctpop.i32(i32 %{{.+}})

unsigned char test_builtin_bitreverse8(unsigned char x) {
  return __builtin_bitreverse8(x);
}

// CIR-LABEL: @_Z24test_builtin_bitreverse8h
// CIR:         %{{.+}} = cir.bitreverse %{{.+}} : !u8i

// LLVM-LABEL: @_Z24test_builtin_bitreverse8h
// LLVM:         %{{.+}} = call i8 @llvm.bitreverse.i8(i8 %{{.+}})

// OGCG-LABEL: @_Z24test_builtin_bitreverse8h
// OGCG:         %{{.+}} = call i8 @llvm.bitreverse.i8(i8 %{{.+}})

unsigned short test_builtin_bitreverse16(unsigned short x) {
  return __builtin_bitreverse16(x);
}

// CIR-LABEL: @_Z25test_builtin_bitreverse16t
// CIR:         %{{.+}} = cir.bitreverse %{{.+}} : !u16i

// LLVM-LABEL: @_Z25test_builtin_bitreverse16t
// LLVM:         %{{.+}} = call i16 @llvm.bitreverse.i16(i16 %{{.+}})

// OGCG-LABEL: @_Z25test_builtin_bitreverse16t
// OGCG:         %{{.+}} = call i16 @llvm.bitreverse.i16(i16 %{{.+}})

unsigned test_builtin_bitreverse32(unsigned x) {
  return __builtin_bitreverse32(x);
}

// CIR-LABEL: @_Z25test_builtin_bitreverse32j
// CIR:         %{{.+}} = cir.bitreverse %{{.+}} : !u32i

// LLVM-LABEL: @_Z25test_builtin_bitreverse32j
// LLVM:         %{{.+}} = call i32 @llvm.bitreverse.i32(i32 %{{.+}})

// OGCG-LABEL: @_Z25test_builtin_bitreverse32j
// OGCG:         %{{.+}} = call i32 @llvm.bitreverse.i32(i32 %{{.+}})

unsigned long long test_builtin_bitreverse64(unsigned long long x) {
  return __builtin_bitreverse64(x);
}

// CIR-LABEL: @_Z25test_builtin_bitreverse64y
// CIR:         %{{.+}} = cir.bitreverse %{{.+}} : !u64i

// LLVM-LABEL: @_Z25test_builtin_bitreverse64y
// LLVM:         %{{.+}} = call i64 @llvm.bitreverse.i64(i64 %{{.+}})

// OGCG-LABEL: @_Z25test_builtin_bitreverse64y
// OGCG:         %{{.+}} = call i64 @llvm.bitreverse.i64(i64 %{{.+}})

unsigned short test_builtin_bswap16(unsigned short x) {
  return __builtin_bswap16(x);
}

// CIR-LABEL: @_Z20test_builtin_bswap16t
// CIR:         %{{.+}} = cir.byte_swap %{{.+}} : !u16i

// LLVM-LABEL: @_Z20test_builtin_bswap16t
// LLVM:         %{{.+}} = call i16 @llvm.bswap.i16(i16 %{{.+}})

// OGCG-LABEL: @_Z20test_builtin_bswap16t
// OGCG:         %{{.+}} = call i16 @llvm.bswap.i16(i16 %{{.+}})

unsigned test_builtin_bswap32(unsigned x) {
  return __builtin_bswap32(x);
}

// CIR-LABEL: @_Z20test_builtin_bswap32j
// CIR:         %{{.+}} = cir.byte_swap %{{.+}} : !u32i

// LLVM-LABEL: @_Z20test_builtin_bswap32j
// LLVM:         %{{.+}} = call i32 @llvm.bswap.i32(i32 %{{.+}})

// OGCG-LABEL: @_Z20test_builtin_bswap32j
// OGCG:         %{{.+}} = call i32 @llvm.bswap.i32(i32 %{{.+}})

unsigned long long test_builtin_bswap64(unsigned long long x) {
  return __builtin_bswap64(x);
}

// CIR-LABEL: @_Z20test_builtin_bswap64y
// CIR:         %{{.+}} = cir.byte_swap %{{.+}} : !u64i

// LLVM-LABEL: @_Z20test_builtin_bswap64y
// LLVM:         %{{.+}} = call i64 @llvm.bswap.i64(i64 %{{.+}})

// OGCG-LABEL: @_Z20test_builtin_bswap64y
// OGCG:         %{{.+}} = call i64 @llvm.bswap.i64(i64 %{{.+}})

unsigned char test_builtin_rotateleft8(unsigned char x, unsigned char y) {
  return __builtin_rotateleft8(x, y);
}

// CIR-LABEL: @_Z24test_builtin_rotateleft8hh
// CIR:         %{{.+}} = cir.rotate left %{{.+}}, %{{.+}} : !u8i

// LLVM-LABEL: @_Z24test_builtin_rotateleft8hh
// LLVM:         %[[INPUT:.+]] = load i8, ptr %{{.+}}, align 1
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i8, ptr %{{.+}}, align 1
// LLVM-NEXT:    %{{.+}} = call i8 @llvm.fshl.i8(i8 %[[INPUT]], i8 %[[INPUT]], i8 %[[AMOUNT]])

// OGCG-LABEL: @_Z24test_builtin_rotateleft8hh
// OGCG:         %[[INPUT:.+]] = load i8, ptr %{{.+}}, align 1
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i8, ptr %{{.+}}, align 1
// OGCG-NEXT:    %{{.+}} = call i8 @llvm.fshl.i8(i8 %[[INPUT]], i8 %[[INPUT]], i8 %[[AMOUNT]])

unsigned short test_builtin_rotateleft16(unsigned short x, unsigned short y) {
  return __builtin_rotateleft16(x, y);
}

// CIR-LABEL: @_Z25test_builtin_rotateleft16tt
// CIR:         %{{.+}} = cir.rotate left %{{.+}}, %{{.+}} : !u16i

// LLVM-LABEL: @_Z25test_builtin_rotateleft16tt
// LLVM:         %[[INPUT:.+]] = load i16, ptr %{{.+}}, align 2
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i16, ptr %{{.+}}, align 2
// LLVM-NEXT:    %{{.+}} = call i16 @llvm.fshl.i16(i16 %[[INPUT]], i16 %[[INPUT]], i16 %[[AMOUNT]])

// OGCG-LABEL: @_Z25test_builtin_rotateleft16tt
// OGCG:         %[[INPUT:.+]] = load i16, ptr %{{.+}}, align 2
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i16, ptr %{{.+}}, align 2
// OGCG-NEXT:    %{{.+}} = call i16 @llvm.fshl.i16(i16 %[[INPUT]], i16 %[[INPUT]], i16 %[[AMOUNT]])

unsigned test_builtin_rotateleft32(unsigned x, unsigned y) {
  return __builtin_rotateleft32(x, y);
}

// CIR-LABEL: @_Z25test_builtin_rotateleft32jj
// CIR:         %{{.+}} = cir.rotate left %{{.+}}, %{{.+}} : !u32i

// LLVM-LABEL: @_Z25test_builtin_rotateleft32jj
// LLVM:         %[[INPUT:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %{{.+}} = call i32 @llvm.fshl.i32(i32 %[[INPUT]], i32 %[[INPUT]], i32 %[[AMOUNT]])

// OGCG-LABEL: @_Z25test_builtin_rotateleft32jj
// OGCG:         %[[INPUT:.+]] = load i32, ptr %{{.+}}, align 4
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i32, ptr %{{.+}}, align 4
// OGCG-NEXT:    %{{.+}} = call i32 @llvm.fshl.i32(i32 %[[INPUT]], i32 %[[INPUT]], i32 %[[AMOUNT]])

unsigned long long test_builtin_rotateleft64(unsigned long long x,
                                             unsigned long long y) {
  return __builtin_rotateleft64(x, y);
}

// CIR-LABEL: @_Z25test_builtin_rotateleft64yy
// CIR:         %{{.+}} = cir.rotate left %{{.+}}, %{{.+}} : !u64i

// LLVM-LABEL: @_Z25test_builtin_rotateleft64yy
// LLVM:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %{{.+}} = call i64 @llvm.fshl.i64(i64 %[[INPUT]], i64 %[[INPUT]], i64 %[[AMOUNT]])

// OGCG-LABEL: @_Z25test_builtin_rotateleft64yy
// OGCG:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %{{.+}} = call i64 @llvm.fshl.i64(i64 %[[INPUT]], i64 %[[INPUT]], i64 %[[AMOUNT]])

unsigned char test_builtin_rotateright8(unsigned char x, unsigned char y) {
  return __builtin_rotateright8(x, y);
}

// CIR-LABEL: @_Z25test_builtin_rotateright8hh
// CIR:         %{{.+}} = cir.rotate right %{{.+}}, %{{.+}} : !u8i

// LLVM-LABEL: @_Z25test_builtin_rotateright8hh
// LLVM:         %[[INPUT:.+]] = load i8, ptr %{{.+}}, align 1
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i8, ptr %{{.+}}, align 1
// LLVM-NEXT:    %{{.+}} = call i8 @llvm.fshr.i8(i8 %[[INPUT]], i8 %[[INPUT]], i8 %[[AMOUNT]])

// OGCG-LABEL: @_Z25test_builtin_rotateright8hh
// OGCG:         %[[INPUT:.+]] = load i8, ptr %{{.+}}, align 1
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i8, ptr %{{.+}}, align 1
// OGCG-NEXT:    %{{.+}} = call i8 @llvm.fshr.i8(i8 %[[INPUT]], i8 %[[INPUT]], i8 %[[AMOUNT]])

unsigned short test_builtin_rotateright16(unsigned short x, unsigned short y) {
  return __builtin_rotateright16(x, y);
}

// CIR-LABEL: @_Z26test_builtin_rotateright16tt
// CIR:         %{{.+}} = cir.rotate right %{{.+}}, %{{.+}} : !u16i

// LLVM-LABEL: @_Z26test_builtin_rotateright16tt
// LLVM:         %[[INPUT:.+]] = load i16, ptr %{{.+}}, align 2
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i16, ptr %{{.+}}, align 2
// LLVM-NEXT:    %{{.+}} = call i16 @llvm.fshr.i16(i16 %[[INPUT]], i16 %[[INPUT]], i16 %[[AMOUNT]])

// OGCG-LABEL: @_Z26test_builtin_rotateright16tt
// OGCG:         %[[INPUT:.+]] = load i16, ptr %{{.+}}, align 2
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i16, ptr %{{.+}}, align 2
// OGCG-NEXT:    %{{.+}} = call i16 @llvm.fshr.i16(i16 %[[INPUT]], i16 %[[INPUT]], i16 %[[AMOUNT]])

unsigned test_builtin_rotateright32(unsigned x, unsigned y) {
  return __builtin_rotateright32(x, y);
}

// CIR-LABEL: @_Z26test_builtin_rotateright32jj
// CIR:         %{{.+}} = cir.rotate right %{{.+}}, %{{.+}} : !u32i

// LLVM-LABEL: @_Z26test_builtin_rotateright32jj
// LLVM:         %[[INPUT:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i32, ptr %{{.+}}, align 4
// LLVM-NEXT:    %{{.+}} = call i32 @llvm.fshr.i32(i32 %[[INPUT]], i32 %[[INPUT]], i32 %[[AMOUNT]])

// OGCG-LABEL: @_Z26test_builtin_rotateright32jj
// OGCG:         %[[INPUT:.+]] = load i32, ptr %{{.+}}, align 4
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i32, ptr %{{.+}}, align 4
// OGCG-NEXT:    %{{.+}} = call i32 @llvm.fshr.i32(i32 %[[INPUT]], i32 %[[INPUT]], i32 %[[AMOUNT]])

unsigned long long test_builtin_rotateright64(unsigned long long x,
                                              unsigned long long y) {
  return __builtin_rotateright64(x, y);
}

// CIR-LABEL: @_Z26test_builtin_rotateright64yy
// CIR:         %{{.+}} = cir.rotate right %{{.+}}, %{{.+}} : !u64i

// LLVM-LABEL: @_Z26test_builtin_rotateright64yy
// LLVM:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %[[AMOUNT:.+]] = load i64, ptr %{{.+}}, align 8
// LLVM-NEXT:    %{{.+}} = call i64 @llvm.fshr.i64(i64 %[[INPUT]], i64 %[[INPUT]], i64 %[[AMOUNT]])

// OGCG-LABEL: @_Z26test_builtin_rotateright64yy
// OGCG:         %[[INPUT:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %[[AMOUNT:.+]] = load i64, ptr %{{.+}}, align 8
// OGCG-NEXT:    %{{.+}} = call i64 @llvm.fshr.i64(i64 %[[INPUT]], i64 %[[INPUT]], i64 %[[AMOUNT]])
