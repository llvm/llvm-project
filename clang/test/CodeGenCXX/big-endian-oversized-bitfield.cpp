// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -emit-llvm -fdump-record-layouts-simple -o %t.be.ll %s | FileCheck %s --check-prefix=LAYOUT
// RUN: FileCheck %s --check-prefix=IR <%t.be.ll
// RUN: %clang_cc1 -triple s390x-linux-gnu -emit-llvm -fdump-record-layouts-simple -o %t.s390x.ll %s | FileCheck %s --check-prefix=LAYOUT
// RUN: FileCheck %s --check-prefix=IR <%t.s390x.ll
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -emit-llvm -fdump-record-layouts-simple -o %t.ppc64.ll %s | FileCheck %s --check-prefix=LAYOUT
// RUN: FileCheck %s --check-prefix=IR <%t.ppc64.ll

// An oversized bit-field has a declared width larger than its type. Only the
// type width is a value; the rest is padding, and the value bits come first.
// Big endian (AArch64, SystemZ, PowerPC) places 0xAB in the high byte of the
// 16-bit container (memory AB 00). A load therefore shifts the container right
// by 8. A store shifts the new value left by 8 and keeps the low byte.

#pragma clang diagnostic ignored "-Wbitfield-width"

struct S {
  unsigned char value : 16;
};

// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: ]>

// First byte is 0xAB. The padding byte is not the value.
// IR: @global = global { i8, i8 } { i8 -85, i8 undef }, align 2
S global = {0xAB};

// IR-LABEL: define {{.*}} @_Z3getPK1S(
// IR: [[P:%.*]] = load ptr, ptr %p.addr
// IR: [[LOAD:%.*]] = load i16, ptr [[P]]
// IR-NEXT: [[SHL:%.*]] = lshr i16 [[LOAD]], 8
// IR-NEXT: trunc i16 [[SHL]] to i8
unsigned char get(const S *p) {
  return p->value;
}

// The value goes in the high byte. The low byte is padding and is preserved.
// IR-LABEL: define {{.*}} @_Z3setP1Sh(
// IR: [[V:%.*]] = load i8, ptr %v.addr
// IR: [[PTR:%.*]] = load ptr, ptr %p.addr
// IR: [[EXT:%.*]] = zext i8 [[V]] to i16
// IR: [[OLD:%.*]] = load i16, ptr [[PTR]]
// IR-NEXT: [[MASKED:%.*]] = and i16 [[EXT]], 255
// IR-NEXT: [[SHIFTED:%.*]] = shl i16 [[MASKED]], 8
// IR-NEXT: [[KEPT:%.*]] = and i16 [[OLD]], 255
// IR-NEXT: [[MERGED:%.*]] = or i16 [[KEPT]], [[SHIFTED]]
// IR-NEXT: store i16 [[MERGED]], ptr [[PTR]]
void set(S *p, unsigned char v) {
  p->value = v;
}
