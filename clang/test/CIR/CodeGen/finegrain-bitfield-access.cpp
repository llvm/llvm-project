// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -ffine-grained-bitfield-accesses %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -ffine-grained-bitfield-accesses %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -ffine-grained-bitfield-accesses %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

struct S1 {
  unsigned f1:2;
  unsigned f2:6;
  unsigned f3:8;
  unsigned f4:4;
  unsigned f5:8;
};

// CIR-DAG: !rec_S1 = !cir.record<struct "S1" {!u8i, !u8i, !u16i}>
// LLVM-DAG: %struct.S1 = type { i8, i8, i16 }
// OGCG-DAG: %struct.S1 = type { i8, i8, i16 }

struct S2 {
  unsigned long f1:16;
  unsigned long f2:16;
  unsigned long f3:6;
};

// CIR-DAG: !rec_S2 = !cir.record<struct "S2" padded {!u16i, !u16i, !u8i, !cir.array<!u8i x 3>}>
// LLVM-DAG: %struct.S2 = type { i16, i16, i8, [3 x i8] }
// OGCG-DAG: %struct.S2 = type { i16, i16, i8, [3 x i8] }

struct S3 {
  unsigned long f1:14;
  unsigned long f2:18;
  unsigned long f3:32;
};

// CIR-DAG: !rec_S3 = !cir.record<struct "S3" {!u32i, !u32i}>
// LLVM-DAG: %struct.S3 = type { i32, i32 }
// OGCG-DAG: %struct.S3 = type { i32, i32 }

S1 a1;
S2 a2;
S3 a3;

unsigned read8_1() {
  return a1.f3;
}

// CIR-LABEL: @_Z7read8_1v
// CIR: [[MEMBER:%.*]] = cir.get_member %1[1] {name = "f3"} : !cir.ptr<!rec_S1> -> !cir.ptr<!u8i>
// CIR: [[BITFI:%.*]] = cir.get_bitfield align(1) (#bfi_f3, [[MEMBER]] : !cir.ptr<!u8i>) -> !u32i
// CIR: cir.store [[BITFI]], {{.*}} : !u32i, !cir.ptr<!u32i>
// CIR: [[RET:%.*]] = cir.load {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.return [[RET]] : !u32i

// LLVM-LABEL: @_Z7read8_1v
// LLVM:  [[MEMBER:%.*]] = load i8, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 1), align 1
// LLVM:  [[BFCAST:%.*]] = zext i8 [[MEMBER]] to i32
// LLVM:  store i32 [[BFCAST]], ptr {{.*}}, align 4
// LLVM:  [[RET:%.*]] = load i32, ptr {{.*}}, align 4
// LLVM:  ret i32 [[RET]]

// OGCG-LABEL: @_Z7read8_1v
// OGCG: [[BFLOAD:%.*]] = load i8, ptr getelementptr inbounds nuw (%struct.S1, ptr {{.*}}, i32 0, i32 1), align 1
// OGCG-NEXT: [[BFCAST:%.*]] = zext i8 [[BFLOAD]] to i32
// OGCG-NEXT: ret i32 [[BFCAST]]

void write8_1() {
  a1.f3 = 3;
}

// CIR-LABEL: @_Z8write8_1v
// CIR: [[CONST3:%.*]] = cir.const #cir.int<3> : !u32i
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[1] {name = "f3"} : !cir.ptr<!rec_S1> -> !cir.ptr<!u8i>
// CIR: cir.set_bitfield align(1) (#bfi_f3, [[MEMBER]] : !cir.ptr<!u8i>, [[CONST3]] : !u32i) -> !u32i

// LLVM-LABEL: @_Z8write8_1v
// LLVM:  store i8 3, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 1), align 1
// LLVM:  ret void

// OGCG-LABEL: @_Z8write8_1v
// OGCG: store i8 3, ptr getelementptr inbounds nuw (%struct.S1, ptr {{.*}}, i32 0, i32 1), align 1
// OGCG-NEXT: ret void

unsigned read8_2() {

  return a1.f5;
}

// CIR-LABEL: @_Z7read8_2v
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[2] {name = "f5"} : !cir.ptr<!rec_S1> -> !cir.ptr<!u16i>
// CIR: [[BITFI:%.*]] = cir.get_bitfield align(2) (#bfi_f5, [[MEMBER]] : !cir.ptr<!u16i>) -> !u32i
// CIR: cir.store [[BITFI]], {{.*}} : !u32i, !cir.ptr<!u32i>
// CIR: [[RET:%.*]] = cir.load {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.return [[RET]] : !u32i

// LLVM-LABEL: @_Z7read8_2v
// LLVM:  [[BFLOAD:%.*]] = load i16, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 2), align 2
// LLVM:  [[BFLSHR:%.*]] = lshr i16 [[BFLOAD]], 4
// LLVM:  [[BFCLEAR:%.*]] = and i16 [[BFLSHR]], 255
// LLVM:  [[BFCAST:%.*]] = zext i16 [[BFCLEAR]] to i32
// LLVM:  store i32 [[BFCAST]], ptr {{.*}}, align 4
// LLVM:  [[RET:%.*]] = load i32, ptr {{.*}}, align 4
// LLVM:  ret i32 [[RET]]

// OGCG-LABEL: @_Z7read8_2v
// OGCG: [[BFLOAD:%.*]] = load i16, ptr getelementptr inbounds nuw (%struct.S1, ptr {{.*}}, i32 0, i32 2), align 2
// OGCG-NEXT: [[BFLSHR:%.*]] = lshr i16 [[BFLOAD]], 4
// OGCG-NEXT: [[BFCLEAR:%.*]] = and i16 [[BFLSHR]], 255
// OGCG-NEXT: [[BFCAST:%.*]] = zext i16 [[BFCLEAR]] to i32
// OGCG-NEXT: ret i32 [[BFCAST]]

void write8_2() {
  a1.f5 = 3;
}

// CIR-LABEL: @_Z8write8_2v
// CIR: [[CONST3:%.*]] = cir.const #cir.int<3> : !u32i
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[2] {name = "f5"} : !cir.ptr<!rec_S1> -> !cir.ptr<!u16i>
// CIR: cir.set_bitfield align(2) (#bfi_f5, [[MEMBER]] : !cir.ptr<!u16i>, [[CONST3]] : !u32i) -> !u32i

// LLVM-LABEL: @_Z8write8_2v
// LLVM:  [[BFLOAD:%.*]] = load i16, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 2), align 2
// LLVM:  [[BFCLEAR:%.*]] = and i16 [[BFLOAD]], -4081
// LLVM:  [[BFSET:%.*]] = or i16 [[BFCLEAR]], 48
// LLVM:  store i16 [[BFSET]], ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 2), align 2
// LLVM:  ret void

// OGCG-LABEL: @_Z8write8_2v
// OGCG: [[BFLOAD:%.*]] = load i16, ptr getelementptr inbounds nuw (%struct.S1, ptr {{.*}}, i32 0, i32 2), align 2
// OGCG-NEXT: [[BFCLEAR:%.*]] = and i16 [[BFLOAD]], -4081
// OGCG-NEXT: [[BFSET:%.*]] = or i16 [[BFCLEAR]], 48
// OGCG-NEXT: store i16 [[BFSET]], ptr getelementptr inbounds nuw (%struct.S1, ptr {{.*}}, i32 0, i32 2), align 2
// OGCG-NEXT: ret void

unsigned read16_1() {
  return a2.f1;
}

// CIR-LABEL: @_Z8read16_1v
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[0] {name = "f1"} : !cir.ptr<!rec_S2> -> !cir.ptr<!u16i>
// CIR: [[BITFI:%.*]] = cir.get_bitfield align(8) (#bfi_f1, [[MEMBER]] : !cir.ptr<!u16i>) -> !u64i
// CIR: [[BFCAST:%.*]] = cir.cast integral [[BITFI]] : !u64i -> !u32i
// CIR: cir.store [[BFCAST]], {{.*}} : !u32i, !cir.ptr<!u32i>
// CIR: [[RET:%.*]] = cir.load {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.return [[RET]] : !u32i

// LLVM-LABEL: @_Z8read16_1v
// LLVM:  [[BFLOAD:%.*]] = load i16, ptr {{.*}}, align 8
// LLVM:  [[BFCAST:%.*]] = zext i16 [[BFLOAD]] to i64
// LLVM:  [[BF:%.*]] = trunc i64 [[BFCAST]] to i32
// LLVM:  store i32 [[BF]], ptr {{.*}}, align 4
// LLVM:  [[RET:%.*]] = load i32, ptr {{.*}}, align 4
// LLVM:  ret i32 [[RET]]

// OGCG-LABEL: @_Z8read16_1v
// OGCG: [[BFLOAD:%.*]] = load i16, ptr {{.*}}, align 8
// OGCG-NEXT: [[BFCAST:%.*]] = zext i16 [[BFLOAD]] to i64
// OGCG-NEXT: [[RET:%.*]] = trunc i64 [[BFCAST]] to i32
// OGCG-NEXT: ret i32 [[RET]]

unsigned read16_2() {
  return a2.f2;
}

// CIR-LABEL: @_Z8read16_2v
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[1] {name = "f2"} : !cir.ptr<!rec_S2> -> !cir.ptr<!u16i>
// CIR: [[BITFI:%.*]] = cir.get_bitfield align(2) (#bfi_f2, [[MEMBER]] : !cir.ptr<!u16i>) -> !u64i
// CIR: [[BFCAST:%.*]] = cir.cast integral [[BITFI]] : !u64i -> !u32i
// CIR: cir.store [[BFCAST]], {{.*}} : !u32i, !cir.ptr<!u32i>
// CIR: [[RET:%.*]] = cir.load {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.return [[RET]] : !u32i

// LLVM-LABEL: @_Z8read16_2v
// LLVM:  [[BFLOAD:%.*]] = load i16, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 2), align 2
// LLVM:  [[BFCAST:%.*]] = zext i16 [[BFLOAD]] to i64
// LLVM:  [[BF:%.*]] = trunc i64 [[BFCAST]] to i32
// LLVM:  store i32 [[BF]], ptr {{.*}}, align 4
// LLVM:  [[RET:%.*]] = load i32, ptr {{.*}}, align 4
// LLVM:  ret i32 [[RET]]

// OGCG-LABEL: @_Z8read16_2v
// OGCG: [[BFLOAD:%.*]] = load i16, ptr getelementptr inbounds nuw (%struct.S2, ptr {{.*}}, i32 0, i32 1), align 2
// OGCG-NEXT: [[BFCAST:%.*]] = zext i16 [[BFLOAD]] to i64
// OGCG-NEXT: [[RET:%.*]] = trunc i64 [[BFCAST]] to i32
// OGCG-NEXT: ret i32 [[RET]]

void write16_1() {
  a2.f1 = 5;
}

// CIR-LABEL: @_Z9write16_1v
// CIR: [[CONST5:%.*]] = cir.const #cir.int<5> : !u64i
// CIR: [[MEMBER:%.*]]  = cir.get_member {{.*}}[0] {name = "f1"} : !cir.ptr<!rec_S2> -> !cir.ptr<!u16i>
// CIR: cir.set_bitfield align(8) (#bfi_f1, [[MEMBER]] : !cir.ptr<!u16i>, [[CONST5]] : !u64i) -> !u64i
// CIR: cir.return

// LLVM-LABEL: @_Z9write16_1v
// LLVM:  store i16 5, ptr {{.*}}, align 8
// LLVM:  ret void

// OGCG-LABEL: @_Z9write16_1v
// OGCG: store i16 5, ptr {{.*}}, align 8
// OGCG-NEXT: ret void

void write16_2() {

  a2.f2 = 5;
}

// CIR-LABEL: @_Z9write16_2v
// CIR: [[CONST5:%.*]] = cir.const #cir.int<5> : !u64i
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[1] {name = "f2"} : !cir.ptr<!rec_S2> -> !cir.ptr<!u16i>
// CIR: cir.set_bitfield align(2) (#bfi_f2, [[MEMBER]] : !cir.ptr<!u16i>, [[CONST5]] : !u64i) -> !u64i
// CIR: cir.return

// LLVM-LABEL: @_Z9write16_2v
// LLVM: store i16 5, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 2), align 2
// LLVM: ret void

// OGCG-LABEL: @_Z9write16_2v
// OGCG: store i16 5, ptr getelementptr inbounds nuw (%struct.S2, ptr {{.*}}, i32 0, i32 1), align 2
// OGCG-NEXT: ret void

unsigned read32_1() {

  return a3.f3;
}
// CIR-LABEL: @_Z8read32_1v
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[1] {name = "f3"} : !cir.ptr<!rec_S3> -> !cir.ptr<!u32i>
// CIR: [[BITFI:%.*]] = cir.get_bitfield align(4) (#bfi_f3_1, [[MEMBER]] : !cir.ptr<!u32i>) -> !u64i
// CIR: [[BFCAST:%.*]] = cir.cast integral [[BITFI]] : !u64i -> !u32i
// CIR: cir.store [[BFCAST]], {{.*}} : !u32i, !cir.ptr<!u32i>
// CIR: [[RET:%.*]] = cir.load {{.*}} : !cir.ptr<!u32i>, !u32i
// CIR: cir.return [[RET]] : !u32i

// LLVM-LABEL: @_Z8read32_1v
// LLVM: [[BFLOAD:%.*]] = load i32, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 4), align 4
// LLVM: [[BFCAST:%.*]] = zext i32 [[BFLOAD]] to i64
// LLVM: [[BF:%.*]] = trunc i64 [[BFCAST]] to i32
// LLVM: store i32 [[BF]], ptr {{.*}}, align 4
// LLVM: [[RET:%.*]] = load i32, ptr {{.*}}, align 4
// LLVM: ret i32 [[RET]]

// OGCG-LABEL: @_Z8read32_1v
// OGCG: [[BFLOAD:%.*]] = load i32, ptr getelementptr inbounds nuw (%struct.S3, ptr {{.*}}, i32 0, i32 1), align 4
// OGCG-NEXT: [[BFCAST:%.*]] = zext i32 %bf.load to i64
// OGCG-NEXT: [[RET:%.*]] = trunc i64 %bf.cast to i32
// OGCG-NEXT: ret i32 [[RET]]

void write32_1() {
  a3.f3 = 5;
}

// CIR-LABEL: @_Z9write32_1v
// CIR: [[CONST5:%.*]] = cir.const #cir.int<5> : !u64i
// CIR: [[MEMBER:%.*]] = cir.get_member {{.*}}[1] {name = "f3"} : !cir.ptr<!rec_S3> -> !cir.ptr<!u32i>
// CIR: cir.set_bitfield align(4) (#bfi_f3_1, [[MEMBER]] : !cir.ptr<!u32i>, [[CONST5]] : !u64i) -> !u64i
// CIR: cir.return

// LLVM-LABEL: @_Z9write32_1v
// LLVM:  store i32 5, ptr getelementptr inbounds nuw (i8, ptr {{.*}}, i64 4), align 4
// LLVM:  ret void

// OGCG-LABEL: @_Z9write32_1v
// OGCG: store i32 5, ptr getelementptr inbounds nuw (%struct.S3, ptr {{.*}}, i32 0, i32 1), align 4
// OGCG-NEXT: ret void
