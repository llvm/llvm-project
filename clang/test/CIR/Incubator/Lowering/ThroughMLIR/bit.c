// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir=core %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

int clz_u16(unsigned short x) {
  return __builtin_clzs(x);
}
// CHECK-LABEL: clz_u16
// CHECK: %[[CTLZ:.+]] = math.ctlz %[[INPUT:.+]] : i16
// CHECK: %[[EXTUI:.+]] = arith.extui %[[CTLZ]] : i16 to i32

int clz_u32(unsigned x) {
  return __builtin_clz(x);
}
// CHECK-LABEL: clz_u32
// CHECK: %[[CTLZ:.+]] = math.ctlz %[[INPUT:.+]] : i32
// CHECK: %[[BITCAST:.+]] = arith.bitcast %[[CTLZ]] : i32 to i32

int clz_u64(unsigned long x) {
  return __builtin_clzl(x);
}
// CHECK-LABEL: clz_u64
// CHECK: %[[CTLZ:.+]] = math.ctlz %[[INPUT:.+]] : i64
// CHECK: %[[TRUNCI:.+]] = arith.trunci %[[CTLZ]] : i64 to i32

int ctz_u16(unsigned short x) {
  return __builtin_ctzs(x);
}
// CHECK-LABEL: ctz_u16
// CHECK: %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i16
// CHECK: %[[EXTUI:.+]] = arith.extui %[[CTTZ]] : i16 to i32

int ctz_u32(unsigned x) {
  return __builtin_ctz(x);
}
// CHECK-LABEL: ctz_u32
// CHECK: %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i32
// CHECK: %[[BITCAST:.+]] = arith.bitcast %[[CTTZ]] : i32 to i32

int ctz_u64(unsigned long x) {
  return __builtin_ctzl(x);
}
// CHECK-LABEL: ctz_u64
// CHECK: %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i64
// CHECK: %[[TRUNCI:.+]] = arith.trunci %[[CTTZ]] : i64 to i32

int popcount_u16(unsigned short x) {
  return __builtin_popcountg(x);
}
// CHECK-LABEL: popcount_u16
// CHECK: %[[CTPOP:.+]] = math.ctpop %{{.+}} : i16
// CHECK-NEXT: %{{.+}} = arith.extui %[[CTPOP]] : i16 to i32

int popcount_u32(unsigned x) {
  return __builtin_popcount(x);
}
// CHECK-LABEL: popcount_u32
// CHECK: %[[CTPOP:.+]] = math.ctpop %{{.+}} : i32
// CHECK-NEXT: %[[BITCAST:.+]] = arith.bitcast %[[CTPOP]] : i32 to i32

int popcount_u64(unsigned long x) {
  return __builtin_popcountl(x);
}
// CHECK-LABEL: popcount_u64
// CHECK: %[[CTPOP:.+]] = math.ctpop %[[INPUT:.+]] : i64
// CHECK-NEXT: %[[TRUNCI:.+]] = arith.trunci %[[CTPOP]] : i64 to i32

int clrsb_s32(int x) {
  return __builtin_clrsb(x);
}
// CHECK-LABEL: clrsb_s32
// CHECK: %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-NEXT: %[[CMP:.+]] = arith.cmpi slt, %[[INPUT:.+]], %[[C0_I32]] : i32
// CHECK-NEXT: %[[C_MINUS1_I32:.+]] = arith.constant -1 : i32
// CHECK-NEXT: %[[XORI:.+]] = arith.xori %[[INPUT]], %[[C_MINUS1_I32]] : i32
// CHECK-NEXT: %[[SELECT:.+]] = arith.select %[[CMP]], %[[XORI]], %[[INPUT]] : i32
// CHECK-NEXT: %[[CTLZ:.+]] = math.ctlz %[[SELECT]] : i32
// CHECK-NEXT: %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[SUBI:.+]] = arith.subi %[[CTLZ]], %[[C1_I32]] : i32

int clrsb_s64(long x) {
  return __builtin_clrsbl(x);
}
// CHECK-LABEL: clrsb_s64
// CHECK: %[[C0_I64:.+]] = arith.constant 0 : i64
// CHECK-NEXT: %[[CMP:.+]] = arith.cmpi slt, %[[INPUT:.+]], %[[C0_I64]] : i64
// CHECK-NEXT: %[[C_MINUS1_I64:.+]] = arith.constant -1 : i64
// CHECK-NEXT: %[[XORI:.+]] = arith.xori %[[INPUT]], %[[C_MINUS1_I64]] : i64
// CHECK-NEXT: %[[SELECT:.+]] = arith.select %[[CMP]], %[[XORI]], %[[INPUT]] : i64
// CHECK-NEXT: %[[CTLZ:.+]] = math.ctlz %[[SELECT]] : i64
// CHECK-NEXT: %[[C1_I64:.+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[SUBI:.+]] = arith.subi %[[CTLZ]], %[[C1_I64]] : i64
// CHECK-NEXT: %[[TRUNCI:.+]] = arith.trunci %[[SUBI]] : i64 to i32

int ffs_s32(int x) {
  return __builtin_ffs(x);
}
// CHECK-LABEL: ffs_s32
// CHECK: %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i32
// CHECK-NEXT: %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[ADDI:.+]] = arith.addi %[[CTTZ]], %[[C1_I32]] : i32
// CHECK-NEXT: %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-NEXT: %[[CMPI:.+]] = arith.cmpi eq, %[[INPUT]], %[[C0_I32]] : i32
// CHECK-NEXT: %[[SELECT:.+]] = arith.select %[[CMPI]], %[[C0_I32]], %[[ADDI]] : i32

int ffs_s64(long x) {
  return __builtin_ffsl(x);
}
// CHECK-LABEL: ffs_s64
// CHECK: %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i64
// CHECK-NEXT: %[[C1_I64:.+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[ADDI:.+]] = arith.addi %[[CTTZ]], %[[C1_I64]] : i64
// CHECK-NEXT: %[[C0_I64:.+]] = arith.constant 0 : i64
// CHECK-NEXT: %[[CMPI:.+]] = arith.cmpi eq, %[[INPUT]], %[[C0_I64]] : i64
// CHECK-NEXT: %[[SELECT:.+]] = arith.select %[[CMPI]], %[[C0_I64]], %[[ADDI]] : i64
// CHECK-NEXT: %[[TRUNCI:.+]] = arith.trunci %[[SELECT]] : i64 to i32

int parity_u32(unsigned x) {
  return __builtin_parity(x);
}
// CHECK-LABEL: parity_u32
// CHECK: %[[CTPOP:.+]] = math.ctpop %[[INPUT:.+]] : i32
// CHECK-NEXT: %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-NEXT: %[[ANDI:.+]] = arith.andi %[[CTPOP]], %[[C1_I32]] : i32
// CHECK-NEXT: %[[BITCAST:.+]] = arith.bitcast %[[ANDI]] : i32 to i32

int parity_u64(unsigned long x) {
  return __builtin_parityl(x);
}
// CHECK-LABEL: func.func @parity_u64(%arg0: i64{{.*}}) -> i32 {
// CHECK: %[[CTPOP:.+]] = math.ctpop %[[INPUT:.+]] : i64
// CHECK-NEXT: %[[C1_I64:.+]] = arith.constant 1 : i64
// CHECK-NEXT: %[[ANDI:.+]] = arith.andi %[[CTPOP]], %[[C1_I64]] : i64
// CHECK-NEXT: %[[TRUNCI:.+]] = arith.trunci %[[ANDI]] : i64 to i32
