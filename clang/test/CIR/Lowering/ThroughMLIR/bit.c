// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fno-clangir-direct-lowering -emit-mlir %s -o %t.mlir
// RUN: FileCheck --input-file=%t.mlir %s

int clz_u16(unsigned short x) {
  return __builtin_clzs(x);   
}
// CHECK: func.func @clz_u16(%arg0: i16{{.*}}) -> i32 {
// CHECK: %[[CTLZ:.+]] = math.ctlz %[[INPUT:.+]] : i16
// CHECK: %[[EXTUI:.+]] = arith.extui %[[CTLZ]] : i16 to i32
// CHECK: }

int clz_u32(unsigned x) {
  return __builtin_clz(x);
}
// CHECK: func.func @clz_u32(%arg0: i32{{.*}}) -> i32 {
// CHECK:   %[[CTLZ:.+]] = math.ctlz %[[INPUT:.+]] : i32
// CHECK:   %[[BITCAST:.+]] = arith.bitcast %[[CTLZ]] : i32 to i32
// CHECK: }

int clz_u64(unsigned long x) {
  return __builtin_clzl(x);
}
// CHECK: func.func @clz_u64(%arg0: i64{{.*}}) -> i32 {
// CHECK:   %[[CTLZ:.+]] = math.ctlz %[[INPUT:.+]] : i64
// CHECK:   %[[TRUNCI:.+]] = arith.trunci %[[CTLZ]] : i64 to i32
// CHECK: }

int ctz_u16(unsigned short x) {
  return __builtin_ctzs(x);   
}
// CHECK: func.func @ctz_u16(%arg0: i16{{.*}}) -> i32 {
// CHECK:   %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i16
// CHECK:   %[[EXTUI:.+]] = arith.extui %[[CTTZ]] : i16 to i32
// CHECK: }

int ctz_u32(unsigned x) {
  return __builtin_ctz(x);   
}
// CHECK: func.func @ctz_u32(%arg0: i32{{.*}}) -> i32 {
// CHECK:   %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i32
// CHECK:   %[[BITCAST:.+]] = arith.bitcast %[[CTTZ]] : i32 to i32
// CHECK: }

int ctz_u64(unsigned long x) {
  return __builtin_ctzl(x);   
}
// CHECK: func.func @ctz_u64(%arg0: i64{{.*}}) -> i32 {
// CHECK:   %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i64
// CHECK:   %[[TRUNCI:.+]] = arith.trunci %[[CTTZ]] : i64 to i32
// CHECK: }

int popcount_u16(unsigned short x) {
  return __builtin_popcount(x);
}
// CHECK: func.func @popcount_u16(%arg0: i16{{.*}}) -> i32 {
// CHECK:   %[[EXTUI:.+]] = arith.extui %[[INPUT:.+]] : i16 to i32
// CHECK:   %[[CTPOP:.+]] = math.ctpop %[[EXTUI]] : i32
// CHECK:   %[[BITCAST:.+]] = arith.bitcast %[[CTPOP]] : i32 to i32
// CHECK: }

int popcount_u32(unsigned x) {
  return __builtin_popcount(x);
}
// CHECK: func.func @popcount_u32(%arg0: i32{{.*}}) -> i32 {
// CHECK:   %[[CTPOP:.+]] = math.ctpop %[[INPUT:.+]] : i32
// CHECK:   %[[BITCAST:.+]] = arith.bitcast %[[CTPOP]] : i32 to i32
// CHECK: }

int popcount_u64(unsigned long x) {
  return __builtin_popcountl(x);
}
// CHECK: func.func @popcount_u64(%arg0: i64{{.*}}) -> i32 {
// CHECK:   %[[CTPOP:.+]] = math.ctpop %[[INPUT:.+]] : i64
// CHECK:   %[[TRUNCI:.+]] = arith.trunci %[[CTPOP]] : i64 to i32
// CHECK: }

int clrsb_s32(int x) {
  return __builtin_clrsb(x);
}
// CHECK: func.func @clrsb_s32(%arg0: i32{{.*}}) -> i32 {
// CHECK:   %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK:   %[[CMP:.+]] = arith.cmpi slt, %[[INPUT:.+]], %[[C0_I32]] : i32
// CHECK:   %[[C_MINUS1_I32:.+]] = arith.constant -1 : i32
// CHECK:   %[[XORI:.+]] = arith.xori %[[INPUT]], %[[C_MINUS1_I32]] : i32
// CHECK:   %[[SELECT:.+]] = arith.select %[[CMP]], %[[XORI]], %[[INPUT]] : i32
// CHECK:   %[[CTLZ:.+]] = math.ctlz %[[SELECT]] : i32
// CHECK:   %[[BITCAST:.+]] = arith.bitcast %[[CTLZ]] : i32 to i32 
// CHECK:   %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK:   %[[SUBI:.+]] = arith.subi %[[BITCAST]], %[[C1_I32]] : i32
// CHECK: }

int clrsb_s64(long x) {
  return __builtin_clrsbl(x);
}
// CHECK: func.func @clrsb_s64(%arg0: i64{{.*}}) -> i32 {
// CHECK:   %[[C0_I64:.+]] = arith.constant 0 : i64
// CHECK:   %[[CMP:.+]] = arith.cmpi slt, %[[INPUT:.+]], %[[C0_I64]] : i64
// CHECK:   %[[C_MINUS1_I64:.+]] = arith.constant -1 : i64
// CHECK:   %[[XORI:.+]] = arith.xori %[[INPUT]], %[[C_MINUS1_I64]] : i64
// CHECK:   %[[SELECT:.+]] = arith.select %[[CMP]], %[[XORI]], %[[INPUT]] : i64
// CHECK:   %[[CTLZ:.+]] = math.ctlz %[[SELECT]] : i64
// CHECK:   %[[TRUNCI:.+]] = arith.trunci %[[CTLZ]] : i64 to i32
// CHECK:   %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK:   %[[SUBI:.+]] = arith.subi %[[TRUNCI]], %[[C1_I32]] : i32
// CHECK: }

int ffs_s32(int x) {
  return __builtin_ffs(x);
}
// CHECK: func.func @ffs_s32(%arg0: i32{{.*}}) -> i32 {
// CHECK:   %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i32
// CHECK-NEXT:   %[[BITCAST:.+]] = arith.bitcast %[[CTTZ]] : i32 to i32
// CHECK-NEXT:   %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:   %[[ADDI:.+]] = arith.addi %[[BITCAST]], %[[C1_I32]] : i32
// CHECK-NEXT:   %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:   %[[CMPI:.+]] = arith.cmpi eq, %[[INPUT]], %[[C0_I32]] : i32
// CHECK-NEXT:   %[[C0_I32_1:.+]] = arith.constant 0 : i32
// CHECK-NEXT:   %[[SELECT:.+]] = arith.select %[[CMPI]], %[[C0_I32_1]], %[[ADDI]] : i32
// CHECK: }

int ffs_s64(long x) {
  return __builtin_ffsl(x);
}
// CHECK: func.func @ffs_s64(%arg0: i64{{.*}}) -> i32 {
// CHECK:   %[[CTTZ:.+]] = math.cttz %[[INPUT:.+]] : i64
// CHECK-NEXT:   %[[TRUNCI:.+]] = arith.trunci %[[CTTZ]] : i64 to i32
// CHECK-NEXT:   %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:   %[[ADDI:.+]] = arith.addi %[[TRUNCI]], %[[C1_I32]] : i32
// CHECK-NEXT:   %[[C0_I64:.+]] = arith.constant 0 : i64
// CHECK-NEXT:   %[[CMPI:.+]] = arith.cmpi eq, %[[INPUT]], %[[C0_I64]] : i64
// CHECK-NEXT:   %[[C0_I32:.+]] = arith.constant 0 : i32
// CHECK-NEXT:   %[[SELECT:.+]] = arith.select %[[CMPI]], %[[C0_I32]], %[[ADDI]] : i32
// CHECK: }

int parity_u32(unsigned x) {
  return __builtin_parity(x);
}
// CHECK: func.func @parity_u32(%arg0: i32{{.*}}) -> i32 {
// CHECK:   %[[CTPOP:.+]] = math.ctpop %[[INPUT:.+]] : i32
// CHECK-NEXT:   %[[C1_I32:.+]] = arith.constant 1 : i32
// CHECK-NEXT:   %[[ANDI:.+]] = arith.andi %[[CTPOP]], %[[C1_I32]] : i32
// CHECK-NEXT:   %[[BITCAST:.+]] = arith.bitcast %[[ANDI]] : i32 to i32
// CHECK: }

int parity_u64(unsigned long x) {
  return __builtin_parityl(x);
}
// CHECK: func.func @parity_u64(%arg0: i64{{.*}}) -> i32 {
// CHECK:   %[[CTPOP:.+]] = math.ctpop %[[INPUT:.+]] : i64
// CHECK-NEXT:   %[[C1_I64:.+]] = arith.constant 1 : i64
// CHECK-NEXT:   %[[ANDI:.+]] = arith.andi %[[CTPOP]], %[[C1_I64]] : i64
// CHECK-NEXT:   %[[TRUNCI:.+]] = arith.trunci %[[ANDI]] : i64 to i32
// CHECK: }