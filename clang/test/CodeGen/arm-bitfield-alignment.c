// RUN: %clang_cc1 -triple arm-none-eabi -fdump-record-layouts-simple -ffreestanding -emit-llvm -o %t %s | FileCheck %s -check-prefixes=LAYOUT,LAYOUT-32
// RUN: FileCheck %s -check-prefixes=IR,IR-32 <%t
// RUN: %clang_cc1 -triple aarch64 -fdump-record-layouts-simple -ffreestanding -emit-llvm -o %t %s | FileCheck %s -check-prefixes=LAYOUT,LAYOUT-64
// RUN: FileCheck %s -check-prefixes=IR,IR-64 <%t

extern struct T {
  int b0 : 8;
  int b1 : 24;
  int b2 : 1;
} g;

int func(void) {
  return g.b1;
}

// IR: @g = external global %struct.T, align 4
// IR-32: %{{.*}} = load i32, ptr @g, align 4
// IR-64: %{{.*}} = load i64, ptr @g, align 4

// LAYOUT-LABEL: LLVMType:%struct.T =
// LAYOUT-32-SAME: type { i32, i8 }
// LAYOUT-64-SAME: type { i64 }
// LAYOUT: BitFields:[
// LAYOUT-32-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-32-NEXT: <CGBitFieldInfo Offset:8 Size:24 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-32-NEXT: <CGBitFieldInfo Offset:0 Size:1 IsSigned:1 StorageSize:8 StorageOffset:4
// LAYOUT-64-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-64-NEXT: <CGBitFieldInfo Offset:8 Size:24 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-64-NEXT: <CGBitFieldInfo Offset:32 Size:1 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-NEXT: ]>
