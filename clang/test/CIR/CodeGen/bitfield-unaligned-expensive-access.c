// Mirrors the cases in clang/test/CodeGen/bitfield-access-unit.c.
//
// RUN: %clang_cc1 -triple=amdgpu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple=amdgpu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple=amdgpu -emit-llvm %s -o -| FileCheck %s --check-prefix=OGCG
// RUN: %clang_cc1 -triple=amdgpu -fclangir -emit-cir -fdump-record-layouts %s -o /dev/null | FileCheck --check-prefix=LAYOUT %s
// RUN: %clang_cc1 -triple=amdgpu -emit-llvm -fdump-record-layouts %s -o /dev/null | FileCheck --check-prefix=OGCG-LAYOUT %s

struct A {
  char a : 7;
  char b : 7;
} a;
// CIR-DAG:  !rec_A = !cir.struct<"A" {bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s8i, 7>]>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s8i, 7>]>}>
// LLVM-DAG: %struct.A = type { i8, i8 }
// OGCG-DAG: %struct.A = type { i8, i8 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"A"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:7 isSigned:1 storageSize:8 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:0 size:7 isSigned:1 storageSize:8 storageOffset:1
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.A =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:1
// OGCG-LAYOUT-NEXT:  ]>

struct __attribute__((aligned(2))) B {
  char a : 7;
  char b : 7;
} b;
// CIR-DAG:  !rec_B = !cir.struct<"B" {bitfield !cir.bitfield<!u16i, [#cir.bitfield_decl<!s8i, 7>, #cir.bitfield_decl<!s8i, 7>]>}>
// LLVM-DAG: %struct.B = type { i16 }
// OGCG-DAG: %struct.B = type { i16 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"B"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:7 isSigned:1 storageSize:16 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:8 size:7 isSigned:1 storageSize:16 storageOffset:0
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.B =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:8 Size:7 IsSigned:1 StorageSize:16 StorageOffset:0
// OGCG-LAYOUT-NEXT:  ]>

struct C {
  int f1;
  char f2;
  char a : 7;
  char b : 7;
} c;
// CIR-DAG:  !rec_C = !cir.struct<"C" {data !s32i, data !s8i, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s8i, 7>]>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s8i, 7>]>}>
// LLVM-DAG: %struct.C = type { i32, i8, i8, i8 }
// OGCG-DAG: %struct.C = type { i32, i8, i8, i8 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"C"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:7 isSigned:1 storageSize:8 storageOffset:5
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:0 size:7 isSigned:1 storageSize:8 storageOffset:6
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.C =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:5
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// OGCG-LAYOUT-NEXT:  ]>

struct __attribute__((packed)) D {
  int f1;
  int a : 8;
  int b : 8;
  char _;
} d;
// CIR-DAG:  !rec_D = !cir.struct<"D" packed {data !s32i, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 8>]>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s32i, 8>]>, data !s8i}>
// LLVM-DAG: %struct.D = type <{ i32, i8, i8, i8 }>
// OGCG-DAG: %struct.D = type <{ i32, i8, i8, i8 }>
// LAYOUT-LABEL:      CIR Type:{{.*}}"D"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:8 isSigned:1 storageSize:8 storageOffset:4
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:0 size:8 isSigned:1 storageSize:8 storageOffset:5
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.D =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:4
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:5
// OGCG-LAYOUT-NEXT:  ]>

struct E {
  char a : 7;
  short b : 13;
  unsigned c : 12;
} e;
// CIR-DAG:  !rec_E = !cir.struct<"E" {bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!s8i, 7>, #cir.bitfield_decl<!s16i, 13>]>, bitfield !cir.bitfield<!u16i, [#cir.bitfield_decl<!u32i, 12>]>}>
// LLVM-DAG: %struct.E = type { i32, i16 }
// OGCG-DAG: %struct.E = type { i32, i16 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"E"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:7 isSigned:1 storageSize:32 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:16 size:13 isSigned:1 storageSize:32 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:c offset:0 size:12 isSigned:0 storageSize:16 storageOffset:4
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.E =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:32 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:16 Size:13 IsSigned:1 StorageSize:32 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// OGCG-LAYOUT-NEXT:  ]>

struct F {
  char a : 7;
  short b : 13;
  unsigned c : 12;
  signed char d : 7;
} f;
// CIR-DAG:  !rec_F = !cir.struct<"F" {bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!s8i, 7>, #cir.bitfield_decl<!s16i, 13>]>, bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!u32i, 12>, #cir.bitfield_decl<!s8i, 7>]>}>
// LLVM-DAG: %struct.F = type { i32, i32 }
// OGCG-DAG: %struct.F = type { i32, i32 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"F"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:7 isSigned:1 storageSize:32 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:16 size:13 isSigned:1 storageSize:32 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:c offset:0 size:12 isSigned:0 storageSize:32 storageOffset:4
// LAYOUT-NEXT:       <CIRBitFieldInfo name:d offset:16 size:7 isSigned:1 storageSize:32 storageOffset:4
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.F =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:32 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:16 Size:13 IsSigned:1 StorageSize:32 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:32 StorageOffset:4
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:16 Size:7 IsSigned:1 StorageSize:32 StorageOffset:4
// OGCG-LAYOUT-NEXT:  ]>

struct G {
  char a : 7;
  short b : 13;
  unsigned c : 12;
  signed char d : 7;
  signed char e;
} g;
// CIR-DAG:  !rec_G = !cir.struct<"G" {bitfield !cir.bitfield<!u32i, [#cir.bitfield_decl<!s8i, 7>, #cir.bitfield_decl<!s16i, 13>]>, bitfield !cir.bitfield<!u16i, [#cir.bitfield_decl<!u32i, 12>]>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s8i, 7>]>, data !s8i}>
// LLVM-DAG: %struct.G = type { i32, i16, i8, i8 }
// OGCG-DAG: %struct.G = type { i32, i16, i8, i8 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"G"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:7 isSigned:1 storageSize:32 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:16 size:13 isSigned:1 storageSize:32 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:c offset:0 size:12 isSigned:0 storageSize:16 storageOffset:4
// LAYOUT-NEXT:       <CIRBitFieldInfo name:d offset:0 size:7 isSigned:1 storageSize:8 storageOffset:6
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.G =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:32 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:16 Size:13 IsSigned:1 StorageSize:32 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:12 IsSigned:0 StorageSize:16 StorageOffset:4
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:8 StorageOffset:6
// OGCG-LAYOUT-NEXT:  ]>

struct __attribute__((aligned(8))) H {
  char a;
  unsigned b : 24;
  unsigned c __attribute__((aligned(8)));
} h;
// CIR-DAG:  !rec_H = !cir.struct<"H" {data !s8i, bitfield !cir.bitfield<!cir.array<!u8i x 3>, [#cir.bitfield_decl<!u32i, 24>]>, pad !cir.array<!u8i x 4>, data !u32i, pad !cir.array<!u8i x 4>}>
// LLVM-DAG: %struct.H = type { i8, [3 x i8], [4 x i8], i32, [4 x i8] }
// OGCG-DAG: %struct.H = type { i8, [3 x i8], [4 x i8], i32, [4 x i8] }
// LAYOUT-LABEL:      CIR Type:{{.*}}"H"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:0 size:24 isSigned:0 storageSize:24 storageOffset:1
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.H =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:24 IsSigned:0 StorageSize:24 StorageOffset:1
// OGCG-LAYOUT-NEXT:  ]>

struct A64 {
  int a : 16;
  short b : 8;
  long c : 16;
  int d : 16;
  signed char e : 8;
} a64;
// CIR-DAG:  !rec_A64 = !cir.struct<"A64" {bitfield !cir.bitfield<!u64i, [#cir.bitfield_decl<!s32i, 16>, #cir.bitfield_decl<!s16i, 8>, #cir.bitfield_decl<!s64i, 16>, #cir.bitfield_decl<!s32i, 16>, #cir.bitfield_decl<!s8i, 8>]>}>
// LLVM-DAG: %struct.A64 = type { i64 }
// OGCG-DAG: %struct.A64 = type { i64 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"A64"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:16 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:16 size:8 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:c offset:24 size:16 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:d offset:40 size:16 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:e offset:56 size:8 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.A64 =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:40 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:56 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  ]>

struct B64 {
  int a : 16;
  short b : 8;
  long c : 16;
  int d : 16;
  signed char e;
} b64;
// CIR-DAG:  !rec_B64 = !cir.struct<"B64" packed {bitfield !cir.bitfield<!u16i, [#cir.bitfield_decl<!s32i, 16>]>, bitfield !cir.bitfield<!u8i, [#cir.bitfield_decl<!s16i, 8>]>, bitfield !cir.bitfield<!u16i, [#cir.bitfield_decl<!s64i, 16>]>, bitfield !cir.bitfield<!u16i, [#cir.bitfield_decl<!s32i, 16>]>, data !s8i}>
// LLVM-DAG: %struct.B64 = type <{ i16, i8, i16, i16, i8 }>
// OGCG-DAG: %struct.B64 = type <{ i16, i8, i16, i16, i8 }>
// LAYOUT-LABEL:      CIR Type:{{.*}}"B64"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:16 isSigned:1 storageSize:16 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:0 size:8 isSigned:1 storageSize:8 storageOffset:2
// LAYOUT-NEXT:       <CIRBitFieldInfo name:c offset:0 size:16 isSigned:1 storageSize:16 storageOffset:3
// LAYOUT-NEXT:       <CIRBitFieldInfo name:d offset:0 size:16 isSigned:1 storageSize:16 storageOffset:5
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.B64 =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:16 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:8 IsSigned:1 StorageSize:8 StorageOffset:2
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:16 StorageOffset:3
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:16 IsSigned:1 StorageSize:16 StorageOffset:5
// OGCG-LAYOUT-NEXT:  ]>

struct C64 {
  int a : 15;
  short b : 8;
  long c : 16;
  int d : 15;
  signed char e : 7;
} c64;
// CIR-DAG:  !rec_C64 = !cir.struct<"C64" {bitfield !cir.bitfield<!u64i, [#cir.bitfield_decl<!s32i, 15>, #cir.bitfield_decl<!s16i, 8>, #cir.bitfield_decl<!s64i, 16>, #cir.bitfield_decl<!s32i, 15>, #cir.bitfield_decl<!s8i, 7>]>}>
// LLVM-DAG: %struct.C64 = type { i64 }
// OGCG-DAG: %struct.C64 = type { i64 }
// LAYOUT-LABEL:      CIR Type:{{.*}}"C64"
// LAYOUT-NEXT:       IsZeroInitializable:1
// LAYOUT-NEXT:       BitFields:[
// LAYOUT-NEXT:       <CIRBitFieldInfo name:a offset:0 size:15 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:b offset:16 size:8 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:c offset:24 size:16 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:d offset:40 size:15 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       <CIRBitFieldInfo name:e offset:56 size:7 isSigned:1 storageSize:64 storageOffset:0
// LAYOUT-NEXT:       ]>
// OGCG-LAYOUT-LABEL: LLVMType:%struct.C64 =
// OGCG-LAYOUT-NEXT:  IsZeroInitializable:1
// OGCG-LAYOUT-NEXT:  BitFields:[
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:0 Size:15 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:16 Size:8 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:24 Size:16 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:40 Size:15 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  <CGBitFieldInfo Offset:56 Size:7 IsSigned:1 StorageSize:64 StorageOffset:0
// OGCG-LAYOUT-NEXT:  ]>
