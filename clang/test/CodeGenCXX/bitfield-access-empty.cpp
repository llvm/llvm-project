// Check if we can merge bitfields across empty members

// Configs that have cheap unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=aarch64-apple-darwin %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=aarch64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=arm-apple-darwin %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT-DWN32 %s
// RUN: %clang_cc1 -triple=arm-none-eabi %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=i686-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=loongarch64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=powerpcle-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=ve-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=wasm32 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=wasm64 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=x86_64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s

// Big Endian
// RUN: %clang_cc1 -triple=powerpc-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=powerpc64-linux-gnu %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=systemz %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s

// Configs that have expensive unaligned access
// Little Endian
// RUN: %clang_cc1 -triple=amdgcn-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=arc-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=bpf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=csky %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=hexagon-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=le64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=loongarch32-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=nvptx-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=riscv32 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=riscv64 %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=spir-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=xcore-none-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s

// Big endian
// RUN: %clang_cc1 -triple=lanai-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=m68k-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=mips-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=mips64-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=sparc-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s
// RUN: %clang_cc1 -triple=tce-elf %s -emit-llvm -o /dev/null -fdump-record-layouts-simple | FileCheck --check-prefixes CHECK,LAYOUT %s

struct Empty {};

struct P1 {
  unsigned a : 16;
  [[no_unique_address]] Empty e;
  unsigned b : 16;
} p1;
// CHECK-LABEL: LLVMType:%struct.P1 =
// LAYOUT-SAME: type { i16, i16 }
// LAYOUT-DWN32-SAME: type { i16, i16 }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.P1 =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:2

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:2
// CHECK-NEXT: ]>

struct P2 {
  unsigned a : 15;
  [[no_unique_address]] Empty e;
  unsigned b : 15;
} p2;
// CHECK-LABEL: LLVMType:%struct.P2 =
// LAYOUT-SAME: type { i16, i16 }
// LAYOUT-DWN32-SAME: type { i16, i16 }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.P2 =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:15 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:15 IsSigned:0 StorageSize:16 StorageOffset:2

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:15 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:15 IsSigned:0 StorageSize:16 StorageOffset:2
// CHECK-NEXT: ]>

struct P3 {
  unsigned a : 16;
  Empty e;
  unsigned b : 16;
} p3;
// CHECK-LABEL: LLVMType:%struct.P3 =
// LAYOUT-SAME: type { i16, %struct.Empty, i16, [2 x i8] }
// LAYOUT-DWN32-SAME: type <{ i16, %struct.Empty, i16 }>
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.P3 =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:4

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:16 StorageOffset:3
// CHECK-NEXT: ]>

struct P4 {
  unsigned : 0;
} p4;
// CHECK-LABEL: LLVMType:%struct.P4 =
// LAYOUT-SAME: type { {{.+}} }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.P4 =
// CHECK: BitFields:[
// CHECK-NEXT: ]>

struct P5 {
  ~P5();
  unsigned : 0;
} p5;
// CHECK-LABEL: LLVMType:%struct.P5 =
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.P5.base = type {}
// CHECK: BitFields:[
// CHECK-NEXT: ]>

struct P6 {
  unsigned a : 16;
  unsigned b : 8;
  [[no_unique_address]] Empty e;
  unsigned c;
} p6;
// CHECK-LABEL: LLVMType:%struct.P6 =
// LAYOUT-SAME: type { i24, i32 }
// LAYOUT-DWN32-SAME: type { i24, i32 }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.P6 =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:0 StorageSize:32 StorageOffset:0

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:32 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:0 StorageSize:32 StorageOffset:0
// CHECK-NEXT: ]>

struct P7 {
  unsigned a : 16;
  unsigned b : 8;
  Empty e;
  unsigned c;
} p7;
// CHECK-LABEL: LLVMType:%struct.P7 =
// LAYOUT-SAME: type { [3 x i8], %struct.Empty, i32 }
// LAYOUT-DWN32-SAME: type { [3 x i8], %struct.Empty, i32 }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.P7 =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:24 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:0 StorageSize:24 StorageOffset:0

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:0 StorageSize:24 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:0 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>
