// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffine-grained-bitfield-accesses \
// RUN:   -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -ffine-grained-bitfield-accesses \
// RUN:   -emit-llvm -fsanitize=address -o - %s | FileCheck %s --check-prefix=SANITIZE
// Check -fsplit-bitfields will be ignored since sanitizer is enabled.

struct S1 {
  unsigned f1:2;
  unsigned f2:6;
  unsigned f3:8;
  unsigned f4:4;
  unsigned f5:8;
};

S1 a1;
unsigned read8_1() {
  // CHECK-LABEL: @_Z7read8_1v
  // CHECK: %bf.load = load i8, i8* getelementptr inbounds (%struct.S1, %struct.S1* @a1, i32 0, i32 1), align 1
  // CHECK-NEXT: %bf.cast = zext i8 %bf.load to i32
  // CHECK-NEXT: ret i32 %bf.cast
  // SANITIZE-LABEL: @_Z7read8_1v
  // SANITIZE: %bf.load = load i32, i32* getelementptr inbounds {{.*}}, align 4
  // SANITIZE: %bf.lshr = lshr i32 %bf.load, 8
  // SANITIZE: %bf.clear = and i32 %bf.lshr, 255
  // SANITIZE: ret i32 %bf.clear
  return a1.f3;
}
void write8_1() {
  // CHECK-LABEL: @_Z8write8_1v
  // CHECK: store i8 3, i8* getelementptr inbounds (%struct.S1, %struct.S1* @a1, i32 0, i32 1), align 1
  // CHECK-NEXT: ret void
  // SANITIZE-LABEL: @_Z8write8_1v
  // SANITIZE: %bf.load = load i32, i32* getelementptr inbounds {{.*}}, align 4
  // SANITIZE-NEXT: %bf.clear = and i32 %bf.load, -65281
  // SANITIZE-NEXT: %bf.set = or i32 %bf.clear, 768
  // SANITIZE-NEXT: store i32 %bf.set, i32* getelementptr inbounds {{.*}}, align 4
  // SANITIZE-NEXT: ret void
  a1.f3 = 3;
}

unsigned read8_2() {
  // CHECK-LABEL: @_Z7read8_2v
  // CHECK: %bf.load = load i16, i16* getelementptr inbounds (%struct.S1, %struct.S1* @a1, i32 0, i32 2), align 2
  // CHECK-NEXT: %bf.lshr = lshr i16 %bf.load, 4
  // CHECK-NEXT: %bf.clear = and i16 %bf.lshr, 255
  // CHECK-NEXT: %bf.cast = zext i16 %bf.clear to i32
  // CHECK-NEXT: ret i32 %bf.cast
  // SANITIZE-LABEL: @_Z7read8_2v
  // SANITIZE: %bf.load = load i32, i32* getelementptr inbounds {{.*}}, align 4
  // SANITIZE-NEXT: %bf.lshr = lshr i32 %bf.load, 20
  // SANITIZE-NEXT: %bf.clear = and i32 %bf.lshr, 255
  // SANITIZE-NEXT: ret i32 %bf.clear
  return a1.f5;
}
void write8_2() {
  // CHECK-LABEL: @_Z8write8_2v
  // CHECK: %bf.load = load i16, i16* getelementptr inbounds (%struct.S1, %struct.S1* @a1, i32 0, i32 2), align 2
  // CHECK-NEXT: %bf.clear = and i16 %bf.load, -4081
  // CHECK-NEXT: %bf.set = or i16 %bf.clear, 48
  // CHECK-NEXT: store i16 %bf.set, i16* getelementptr inbounds (%struct.S1, %struct.S1* @a1, i32 0, i32 2), align 2
  // CHECK-NEXT: ret void
  // SANITIZE-LABEL: @_Z8write8_2v
  // SANITIZE: %bf.load = load i32, i32* getelementptr inbounds {{.*}}, align 4
  // SANITIZE-NEXT: %bf.clear = and i32 %bf.load, -267386881
  // SANITIZE-NEXT: %bf.set = or i32 %bf.clear, 3145728
  // SANITIZE-NEXT: store i32 %bf.set, i32* getelementptr inbounds {{.*}}, align 4
  // SANITIZE-NEXT: ret void
  a1.f5 = 3;
}

struct S2 {
  unsigned long f1:16;
  unsigned long f2:16;
  unsigned long f3:6;
};

S2 a2;
unsigned read16_1() {
  // CHECK-LABEL: @_Z8read16_1v
  // CHECK: %bf.load = load i16, i16* getelementptr inbounds (%struct.S2, %struct.S2* @a2, i32 0, i32 0), align 8
  // CHECK-NEXT: %bf.cast = zext i16 %bf.load to i64
  // CHECK-NEXT: %conv = trunc i64 %bf.cast to i32
  // CHECK-NEXT: ret i32 %conv
  // SANITIZE-LABEL: @_Z8read16_1v
  // SANITIZE: %bf.load = load i64, i64* bitcast {{.*}}, align 8
  // SANITIZE-NEXT: %bf.clear = and i64 %bf.load, 65535
  // SANITIZE-NEXT: %conv = trunc i64 %bf.clear to i32
  // SANITIZE-NEXT: ret i32 %conv
  return a2.f1;
}
unsigned read16_2() {
  // CHECK-LABEL: @_Z8read16_2v
  // CHECK: %bf.load = load i16, i16* getelementptr inbounds (%struct.S2, %struct.S2* @a2, i32 0, i32 1), align 2
  // CHECK-NEXT: %bf.cast = zext i16 %bf.load to i64
  // CHECK-NEXT: %conv = trunc i64 %bf.cast to i32
  // CHECK-NEXT: ret i32 %conv
  // SANITIZE-LABEL: @_Z8read16_2v
  // SANITIZE: %bf.load = load i64, i64* bitcast {{.*}}, align 8
  // SANITIZE-NEXT: %bf.lshr = lshr i64 %bf.load, 16
  // SANITIZE-NEXT: %bf.clear = and i64 %bf.lshr, 65535
  // SANITIZE-NEXT: %conv = trunc i64 %bf.clear to i32
  // SANITIZE-NEXT: ret i32 %conv
  return a2.f2;
}

void write16_1() {
  // CHECK-LABEL: @_Z9write16_1v
  // CHECK: store i16 5, i16* getelementptr inbounds (%struct.S2, %struct.S2* @a2, i32 0, i32 0), align 8
  // CHECK-NEXT: ret void
  // SANITIZE-LABEL: @_Z9write16_1v
  // SANITIZE: %bf.load = load i64, i64* bitcast {{.*}}, align 8
  // SANITIZE-NEXT: %bf.clear = and i64 %bf.load, -65536
  // SANITIZE-NEXT: %bf.set = or i64 %bf.clear, 5
  // SANITIZE-NEXT: store i64 %bf.set, i64* bitcast {{.*}}, align 8
  // SANITIZE-NEXT: ret void
  a2.f1 = 5;
}
void write16_2() {
  // CHECK-LABEL: @_Z9write16_2v
  // CHECK: store i16 5, i16* getelementptr inbounds (%struct.S2, %struct.S2* @a2, i32 0, i32 1), align 2
  // CHECK-NEXT: ret void
  // SANITIZE-LABEL: @_Z9write16_2v
  // SANITIZE: %bf.load = load i64, i64* bitcast {{.*}}, align 8
  // SANITIZE-NEXT: %bf.clear = and i64 %bf.load, -4294901761
  // SANITIZE-NEXT: %bf.set = or i64 %bf.clear, 327680
  // SANITIZE-NEXT: store i64 %bf.set, i64* bitcast {{.*}}, align 8
  // SANITIZE-NEXT: ret void
  a2.f2 = 5;
}

struct S3 {
  unsigned long f1:14;
  unsigned long f2:18;
  unsigned long f3:32;
};

S3 a3;
unsigned read32_1() {
  // CHECK-LABEL: @_Z8read32_1v
  // CHECK: %bf.load = load i32, i32* getelementptr inbounds (%struct.S3, %struct.S3* @a3, i32 0, i32 1), align 4
  // CHECK-NEXT: %bf.cast = zext i32 %bf.load to i64
  // CHECK-NEXT: %conv = trunc i64 %bf.cast to i32
  // CHECK-NEXT: ret i32 %conv
  // SANITIZE-LABEL: @_Z8read32_1v
  // SANITIZE: %bf.load = load i64, i64* getelementptr inbounds {{.*}}, align 8
  // SANITIZE-NEXT: %bf.lshr = lshr i64 %bf.load, 32
  // SANITIZE-NEXT: %conv = trunc i64 %bf.lshr to i32
  // SANITIZE-NEXT: ret i32 %conv
  return a3.f3;
}
void write32_1() {
  // CHECK-LABEL: @_Z9write32_1v
  // CHECK: store i32 5, i32* getelementptr inbounds (%struct.S3, %struct.S3* @a3, i32 0, i32 1), align 4
  // CHECK-NEXT: ret void
  // SANITIZE-LABEL: @_Z9write32_1v
  // SANITIZE: %bf.load = load i64, i64* getelementptr inbounds {{.*}}, align 8
  // SANITIZE-NEXT: %bf.clear = and i64 %bf.load, 4294967295
  // SANITIZE-NEXT: %bf.set = or i64 %bf.clear, 21474836480
  // SANITIZE-NEXT: store i64 %bf.set, i64* getelementptr inbounds {{.*}}, align 8
  // SANITIZE-NEXT: ret void
  a3.f3 = 5;
}
