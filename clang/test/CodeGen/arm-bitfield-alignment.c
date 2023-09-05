// RUN: %clang_cc1 -triple arm-none-eabi -fdump-record-layouts-simple -ffreestanding -emit-llvm -o %t %s | FileCheck %s -check-prefix=LAYOUT
// RUN: FileCheck %s -check-prefix=IR <%t
// RUN: %clang_cc1 -triple aarch64 -fdump-record-layouts-simple -ffreestanding -emit-llvm -o %t %s | FileCheck %s -check-prefix=LAYOUT
// RUN: FileCheck %s -check-prefix=IR <%t

extern struct T {
  int b0 : 8;
  int b1 : 24;
  int b2 : 1;
} g;

int func(void) {
  return g.b1;
}

// IR: @g = external global %struct.T, align 4
// IR: %{{.*}} = load i64, ptr @g, align 4

// LAYOUT-LABEL: LLVMType:%struct.T =
// LAYOUT-SAME: type { i40 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:8 Size:24 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:32 Size:1 IsSigned:1 StorageSize:64 StorageOffset:0
// LAYOUT-NEXT: ]>
