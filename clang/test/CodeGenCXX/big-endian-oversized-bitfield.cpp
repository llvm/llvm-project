// RUN: %clang_cc1 -triple aarch64_be-linux-gnu -std=c++17 -emit-llvm -O0 \
// RUN:   -fdump-record-layouts-simple -o %t.be.ll %s | FileCheck %s --check-prefix=LAYOUT-BE
// RUN: FileCheck %s --check-prefix=BE <%t.be.ll
// RUN: %clang_cc1 -triple s390x-linux-gnu -std=c++17 -emit-llvm -O0 \
// RUN:   -fdump-record-layouts-simple -o %t.s390x.ll %s | FileCheck %s --check-prefix=LAYOUT-BE
// RUN: FileCheck %s --check-prefix=BE <%t.s390x.ll
// RUN: %clang_cc1 -triple powerpc64-linux-gnu -std=c++17 -emit-llvm -O0 \
// RUN:   -fdump-record-layouts-simple -o %t.ppc64.ll %s | FileCheck %s --check-prefix=LAYOUT-BE
// RUN: FileCheck %s --check-prefix=BE <%t.ppc64.ll

// An oversized bit-field has a declared width larger than its type. Only the
// type width is a value; the rest is padding, and the value bits come first.
// Big endian (AArch64, SystemZ, PowerPC) places 0xAB in the high byte of the
// 16-bit container (memory AB 00). A load therefore shifts the container right
// by 8.

#pragma clang diagnostic ignored "-Wbitfield-width"

struct S {
  unsigned char value : 16;
};

// LAYOUT-BE: BitFields:[
// LAYOUT-BE-NEXT: <CGBitFieldInfo Offset:8 Size:8 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-BE-NEXT: ]>

// First byte is 0xAB. The padding byte is not the value.
// BE: @global = global { i8, i8 } { i8 -85, i8 undef }, align 2
extern "C" S global = {0xAB};

// BE-LABEL: define {{.*}} @get(
// BE: [[P:%.*]] = load ptr, ptr %p.addr
// BE: [[LOAD:%.*]] = load i16, ptr [[P]]
// BE-NEXT: [[SHL:%.*]] = lshr i16 [[LOAD]], 8
// BE-NEXT: trunc i16 [[SHL]] to i8
extern "C" unsigned char get(const S *p) {
  return p->value;
}
