// Check we use tail padding if it is known to be safe

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

// Can use tail padding
struct Pod {
  int a : 16;
  int b : 8;
} P;
// CHECK-LABEL: LLVMType:%struct.Pod =
// LAYOUT-SAME: type { i24 }
// LAYOUT-DWN32-SAME: type { [3 x i8] }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.Pod =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:32 StorageOffset:0

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>

// No tail padding
struct __attribute__((packed)) PPod {
  int a : 16;
  int b : 8;
} PP;
// CHECK-LABEL: LLVMType:%struct.PPod =
// LAYOUT-SAME: type { [3 x i8] }
// LAYOUT-DWN32-SAME: type { [3 x i8] }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.PPod =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:24 StorageOffset:0

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>

// Cannot use tail padding
struct NonPod {
  ~NonPod();
  int a : 16;
  int b : 8;
} NP;
// CHECK-LABEL: LLVMType:%struct.NonPod =
// LAYOUT-SAME: type { [3 x i8], i8 }
// LAYOUT-DWN32-SAME: type { [3 x i8] }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.
// LAYOUT-SAME: NonPod.base = type { [3 x i8] }
// LAYOUT-DWN32-SAME: NonPod = type { [3 x i8] }
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:24 StorageOffset:0

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>

// No tail padding
struct __attribute__((packed)) PNonPod {
  ~PNonPod();
  int a : 16;
  int b : 8;
} PNP;
// CHECK-LABEL: LLVMType:%struct.PNonPod =
// LAYOUT-SAME: type { [3 x i8] }
// LAYOUT-DWN32-SAME: type { [3 x i8] }
// CHECK-NEXT: NonVirtualBaseLLVMType:%struct.PNonPod =
// CHECK: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:24 StorageOffset:0

// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:16 IsSigned:1 StorageSize:24 StorageOffset:0
// LAYOUT-DWN32-NEXT: <CGBitFieldInfo Offset:{{[0-9]+}} Size:8 IsSigned:1 StorageSize:24 StorageOffset:0
// CHECK-NEXT: ]>
