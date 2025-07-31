// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir -fdump-record-layouts %s -o %t.cir 1> %t.cirlayout
// RUN: FileCheck --input-file=%t.cirlayout %s --check-prefix=CIR-LAYOUT

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fdump-record-layouts %s -o %t.ll 1> %t.ogcglayout
// RUN: FileCheck --input-file=%t.ogcglayout %s --check-prefix=OGCG-LAYOUT

typedef struct  {
    unsigned int a : 9;
    volatile unsigned int b : 1;
    unsigned int c : 1;
} st1;

// CIR-LAYOUT:  BitFields:[
// CIR-LAYOUT-NEXT:    <CIRBitFieldInfo name:a offset:0 size:9 isSigned:0 storageSize:16 storageOffset:0 volatileOffset:0 volatileStorageSize:32 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:    <CIRBitFieldInfo name:b offset:9 size:1 isSigned:0 storageSize:16 storageOffset:0 volatileOffset:9 volatileStorageSize:32 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:    <CIRBitFieldInfo name:c offset:10 size:1 isSigned:0 storageSize:16 storageOffset:0 volatileOffset:10 volatileStorageSize:32 volatileStorageOffset:0>

// OGCG-LAYOUT:  BitFields:[
// OGCG-LAYOUT-NEXT:    <CGBitFieldInfo Offset:0 Size:9 IsSigned:0 StorageSize:16 StorageOffset:0 VolatileOffset:0 VolatileStorageSize:32 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:    <CGBitFieldInfo Offset:9 Size:1 IsSigned:0 StorageSize:16 StorageOffset:0 VolatileOffset:9 VolatileStorageSize:32 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:    <CGBitFieldInfo Offset:10 Size:1 IsSigned:0 StorageSize:16 StorageOffset:0 VolatileOffset:10 VolatileStorageSize:32 VolatileStorageOffset:0>

// different base types
typedef struct{
    volatile  short a : 3;
    volatile  int b: 13;
    volatile  long c : 5;
} st2;

// CIR-LAYOUT: BitFields:[
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:a offset:0 size:3 isSigned:1 storageSize:32 storageOffset:0 volatileOffset:0 volatileStorageSize:16 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:b offset:3 size:13 isSigned:1 storageSize:32 storageOffset:0 volatileOffset:3 volatileStorageSize:32 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:c offset:16 size:5 isSigned:1 storageSize:32 storageOffset:0 volatileOffset:16 volatileStorageSize:64 volatileStorageOffset:0>

// OGCG-LAYOUT: BitFields:[
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:0 Size:3 IsSigned:1 StorageSize:32 StorageOffset:0 VolatileOffset:0 VolatileStorageSize:16 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:3 Size:13 IsSigned:1 StorageSize:32 StorageOffset:0 VolatileOffset:3 VolatileStorageSize:32 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:16 Size:5 IsSigned:1 StorageSize:32 StorageOffset:0 VolatileOffset:16 VolatileStorageSize:64 VolatileStorageOffset:0>

typedef struct{
    volatile unsigned int a : 3;
    unsigned int : 0; // zero-length bit-field force next field to aligned int boundary
    volatile unsigned int b : 5;
} st3;

// CIR-LAYOUT: BitFields:[
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:a offset:0 size:3 isSigned:0 storageSize:8 storageOffset:0 volatileOffset:0 volatileStorageSize:32 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:b offset:0 size:5 isSigned:0 storageSize:8 storageOffset:4 volatileOffset:0 volatileStorageSize:0 volatileStorageOffset:0>

// OGCG-LAYOUT: BitFields:[
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:0 Size:3 IsSigned:0 StorageSize:8 StorageOffset:0 VolatileOffset:0 VolatileStorageSize:32 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:0 Size:5 IsSigned:0 StorageSize:8 StorageOffset:4 VolatileOffset:0 VolatileStorageSize:0 VolatileStorageOffset:0>

typedef struct{
    volatile unsigned int a : 3;
    unsigned int z: 2;
    volatile unsigned int b : 5;
} st4;

// CIR-LAYOUT: BitFields:[
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:a offset:0 size:3 isSigned:0 storageSize:16 storageOffset:0 volatileOffset:0 volatileStorageSize:32 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:z offset:3 size:2 isSigned:0 storageSize:16 storageOffset:0 volatileOffset:3 volatileStorageSize:32 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:b offset:5 size:5 isSigned:0 storageSize:16 storageOffset:0 volatileOffset:5 volatileStorageSize:32 volatileStorageOffset:0>

// OGCG-LAYOUT: BitFields:[
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:0 Size:3 IsSigned:0 StorageSize:16 StorageOffset:0 VolatileOffset:0 VolatileStorageSize:32 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:3 Size:2 IsSigned:0 StorageSize:16 StorageOffset:0 VolatileOffset:3 VolatileStorageSize:32 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:5 Size:5 IsSigned:0 StorageSize:16 StorageOffset:0 VolatileOffset:5 VolatileStorageSize:32 VolatileStorageOffset:0>

st1 s1;
st2 s2;
st3 s3;
st4 s4;
