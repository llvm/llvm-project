// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-cir -fdump-record-layouts %s -o %t.cir > %t.cirlayout
// RUN: FileCheck --input-file=%t.cirlayout %s --check-prefix=CIR-LAYOUT
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM

// RUN: %clang_cc1 -triple aarch64-unknown-linux-gnu -emit-llvm -fdump-record-layouts %s -o %t.ll > %t.ogcglayout
// RUN: FileCheck --input-file=%t.ogcglayout %s --check-prefix=OGCG-LAYOUT
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

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
    unsigned int z;
    volatile unsigned long b : 16;
} st4;

// CIR-LAYOUT: BitFields:[
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:a offset:0 size:3 isSigned:0 storageSize:8 storageOffset:0 volatileOffset:0 volatileStorageSize:32 volatileStorageOffset:0>
// CIR-LAYOUT-NEXT:   <CIRBitFieldInfo name:b offset:0 size:16 isSigned:0 storageSize:16 storageOffset:8 volatileOffset:0 volatileStorageSize:64 volatileStorageOffset:1>

// OGCG-LAYOUT: BitFields:[
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:0 Size:3 IsSigned:0 StorageSize:8 StorageOffset:0 VolatileOffset:0 VolatileStorageSize:32 VolatileStorageOffset:0>
// OGCG-LAYOUT-NEXT:   <CGBitFieldInfo Offset:0 Size:16 IsSigned:0 StorageSize:16 StorageOffset:8 VolatileOffset:0 VolatileStorageSize:64 VolatileStorageOffset:1>


void def () {
  st1 s1;
  st2 s2;
  st3 s3;
  st4 s4;
}

int check_load(st1 *s1) {
  return s1->b;
}

// CIR:  cir.func dso_local @check_load
// CIR:    [[LOAD:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.ptr<!rec_st1>>, !cir.ptr<!rec_st1>
// CIR:    [[MEMBER:%.*]] = cir.get_member [[LOAD]][0] {name = "b"} : !cir.ptr<!rec_st1> -> !cir.ptr<!u16i>
// CIR:    [[BITFI:%.*]] = cir.get_bitfield align(4) (#bfi_b, [[MEMBER]] {is_volatile} : !cir.ptr<!u16i>) -> !u32i
// CIR:    [[CAST:%.*]] = cir.cast integral [[BITFI]] : !u32i -> !s32i
// CIR:    cir.store [[CAST]], [[RETVAL:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[RET:%.*]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.return [[RET]] : !s32i

// LLVM:define dso_local i32 @check_load
// LLVM:  [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM:  [[MEMBER:%.*]] = getelementptr %struct.st1, ptr [[LOAD]], i32 0, i32 0
// LLVM:  [[LOADVOL:%.*]] = load volatile i32, ptr [[MEMBER]], align 4
// LLVM:  [[LSHR:%.*]] = lshr i32 [[LOADVOL]], 9
// LLVM:  [[CLEAR:%.*]] = and i32 [[LSHR]], 1
// LLVM:  store i32 [[CLEAR]], ptr [[RETVAL:%.*]], align 4
// LLVM:  [[RET:%.*]] = load i32, ptr [[RETVAL]], align 4
// LLVM:  ret i32 [[RET]]

// OGCG: define dso_local i32 @check_load
// OGCG:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// OGCG:   [[LOADVOL:%.*]] = load volatile i32, ptr [[LOAD]], align 4
// OGCG:   [[LSHR:%.*]] = lshr i32 [[LOADVOL]], 9
// OGCG:   [[CLEAR:%.*]] = and i32 [[LSHR]], 1
// OGCG:   ret i32 [[CLEAR]]

// this volatile bit-field container overlaps with a zero-length bit-field,
// so it may be accessed without using the container's width.
int check_load_exception(st3 *s3) {
  return s3->b;
}

// CIR:  cir.func dso_local @check_load_exception
// CIR:    [[LOAD:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.ptr<!rec_st3>>, !cir.ptr<!rec_st3>
// CIR:    [[MEMBER:%.*]] = cir.get_member [[LOAD]][2] {name = "b"} : !cir.ptr<!rec_st3> -> !cir.ptr<!u8i>
// CIR:    [[BITFI:%.*]] = cir.get_bitfield align(4) (#bfi_b1, [[MEMBER]] {is_volatile} : !cir.ptr<!u8i>) -> !u32i
// CIR:    [[CAST:%.*]] = cir.cast integral [[BITFI]] : !u32i -> !s32i
// CIR:    cir.store [[CAST]], [[RETVAL:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[RET:%.*]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.return [[RET]] : !s32i

// LLVM:define dso_local i32 @check_load_exception
// LLVM:  [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM:  [[MEMBER:%.*]] = getelementptr %struct.st3, ptr [[LOAD]], i32 0, i32 2
// LLVM:  [[LOADVOL:%.*]] = load volatile i8, ptr [[MEMBER]], align 4
// LLVM:  [[CLEAR:%.*]] = and i8 [[LOADVOL]], 31
// LLVM:  [[CAST:%.*]] = zext i8 [[CLEAR]] to i32
// LLVM:  store i32 [[CAST]], ptr [[RETVAL:%.*]], align 4
// LLVM:  [[RET:%.*]] = load i32, ptr [[RETVAL]], align 4
// LLVM:  ret i32 [[RET]]

// OGCG: define dso_local i32 @check_load_exception
// OGCG:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// OGCG:   [[MEMBER:%.*]] = getelementptr inbounds nuw %struct.st3, ptr [[LOAD]], i32 0, i32 2
// OGCG:   [[LOADVOL:%.*]] = load volatile i8, ptr [[MEMBER]], align 4
// OGCG:   [[CLEAR:%.*]] = and i8 [[LOADVOL]], 31
// OGCG:   [[CAST:%.*]] = zext i8 [[CLEAR]] to i32
// OGCG:   ret i32 [[CAST]]

typedef struct {
    volatile int a : 24;
    char b;
    volatile int c: 30;
 } clip;

int clip_load_exception2(clip *c) {
  return c->a;
}

// CIR:  cir.func dso_local @clip_load_exception2
// CIR:    [[LOAD:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.ptr<!rec_clip>>, !cir.ptr<!rec_clip>
// CIR:    [[MEMBER:%.*]] = cir.get_member [[LOAD]][0] {name = "a"} : !cir.ptr<!rec_clip> -> !cir.ptr<!cir.array<!u8i x 3>>
// CIR:    [[BITFI:%.*]] = cir.get_bitfield align(4) (#bfi_a1, [[MEMBER]] {is_volatile} : !cir.ptr<!cir.array<!u8i x 3>>) -> !s32i
// CIR:    cir.store [[BITFI]], [[RETVAL:%.*]] : !s32i, !cir.ptr<!s32i>
// CIR:    [[RET:%.*]] = cir.load [[RETVAL]] : !cir.ptr<!s32i>, !s32i
// CIR:    cir.return [[RET]] : !s32i

// LLVM:define dso_local i32 @clip_load_exception2
// LLVM:  [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM:  [[MEMBER:%.*]] = getelementptr %struct.clip, ptr [[LOAD]], i32 0, i32 0
// LLVM:  [[LOADVOL:%.*]] = load volatile i24, ptr [[MEMBER]], align 4
// LLVM:  [[CAST:%.*]] = sext i24 [[LOADVOL]] to i32
// LLVM:  store i32 [[CAST]], ptr [[RETVAL:%.*]], align 4
// LLVM:  [[RET:%.*]] = load i32, ptr [[RETVAL]], align 4
// LLVM:  ret i32 [[RET]]

// OGCG: define dso_local i32 @clip_load_exception2
// OGCG:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// OGCG:   [[LOADVOL:%.*]] = load volatile i24, ptr [[LOAD]], align 4
// OGCG:   [[CAST:%.*]] = sext i24 [[LOADVOL]] to i32
// OGCG:   ret i32 [[CAST]]

void check_store(st2 *s2) {
  s2->a = 1;
}

// CIR:  cir.func dso_local @check_store
// CIR:    [[CONST:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:    [[CAST:%.*]] = cir.cast integral [[CONST]] : !s32i -> !s16i
// CIR:    [[LOAD:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.ptr<!rec_st2>>, !cir.ptr<!rec_st2>
// CIR:    [[MEMBER:%.*]] = cir.get_member [[LOAD]][0] {name = "a"} : !cir.ptr<!rec_st2> -> !cir.ptr<!u32i>
// CIR:    [[SETBF:%.*]] = cir.set_bitfield align(8) (#bfi_a, [[MEMBER]] : !cir.ptr<!u32i>, [[CAST]] : !s16i) {is_volatile} -> !s16i
// CIR:    cir.return

// LLVM:define dso_local void @check_store
// LLVM:  [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM:  [[MEMBER:%.*]] = getelementptr %struct.st2, ptr [[LOAD]], i32 0, i32 0
// LLVM:  [[LOADVOL:%.*]] = load volatile i16, ptr [[MEMBER]], align 8
// LLVM:  [[CLEAR:%.*]] = and i16 [[LOADVOL]], -8
// LLVM:  [[SET:%.*]] = or i16 [[CLEAR]], 1
// LLVM:  store volatile i16 [[SET]], ptr [[MEMBER]], align 8
// LLVM:  ret void

// OGCG: define dso_local void @check_store
// OGCG:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// OGCG:   [[LOADVOL:%.*]] = load volatile i16, ptr [[LOAD]], align 8
// OGCG:   [[CLEAR:%.*]] = and i16 [[LOADVOL]], -8
// OGCG:   [[SET:%.*]] = or i16 [[CLEAR]], 1
// OGCG:   store volatile i16 [[SET]], ptr [[LOAD]], align 8
// OGCG:   ret void

// this volatile bit-field container overlaps with a zero-length bit-field,
// so it may be accessed without using the container's width.
void check_store_exception(st3 *s3) {
  s3->b = 2;
}

// CIR:  cir.func dso_local @check_store_exception
// CIR:    [[CONST:%.*]] = cir.const #cir.int<2> : !s32i
// CIR:    [[CAST:%.*]] = cir.cast integral [[CONST]] : !s32i -> !u32i
// CIR:    [[LOAD:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.ptr<!rec_st3>>, !cir.ptr<!rec_st3>
// CIR:    [[MEMBER:%.*]] = cir.get_member [[LOAD]][2] {name = "b"} : !cir.ptr<!rec_st3> -> !cir.ptr<!u8i>
// CIR:    [[SETBF:%.*]] = cir.set_bitfield align(4) (#bfi_b1, [[MEMBER]] : !cir.ptr<!u8i>, [[CAST]] : !u32i) {is_volatile} -> !u32i
// CIR:    cir.return

// LLVM:define dso_local void @check_store_exception
// LLVM:  [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM:  [[MEMBER:%.*]] = getelementptr %struct.st3, ptr [[LOAD]], i32 0, i32 2
// LLVM:  [[LOADVOL:%.*]] = load volatile i8, ptr [[MEMBER]], align 4
// LLVM:  [[CLEAR:%.*]] = and i8 [[LOADVOL]], -32
// LLVM:  [[SET:%.*]] = or i8 [[CLEAR]], 2
// LLVM:  store volatile i8 [[SET]], ptr [[MEMBER]], align 4
// LLVM:  ret void

// OGCG: define dso_local void @check_store_exception
// OGCG:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// OGCG:   [[MEMBER:%.*]] = getelementptr inbounds nuw %struct.st3, ptr [[LOAD]], i32 0, i32 2
// OGCG:   [[LOADVOL:%.*]] = load volatile i8, ptr [[MEMBER]], align 4
// OGCG:   [[CLEAR:%.*]] = and i8 [[LOADVOL]], -32
// OGCG:   [[SET:%.*]] = or i8 [[CLEAR]], 2
// OGCG:   store volatile i8 [[SET]], ptr [[MEMBER]], align 4
// OGCG:   ret void

void clip_store_exception2(clip *c) {
  c->a = 3;
}

// CIR:  cir.func dso_local @clip_store_exception2
// CIR:    [[CONST:%.*]] = cir.const #cir.int<3> : !s32i
// CIR:    [[LOAD:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.ptr<!rec_clip>>, !cir.ptr<!rec_clip>
// CIR:    [[MEMBER:%.*]] = cir.get_member [[LOAD]][0] {name = "a"} : !cir.ptr<!rec_clip> -> !cir.ptr<!cir.array<!u8i x 3>>
// CIR:    [[SETBF:%.*]] = cir.set_bitfield align(4) (#bfi_a1, [[MEMBER]] : !cir.ptr<!cir.array<!u8i x 3>>, [[CONST]] : !s32i) {is_volatile} -> !s32i
// CIR:    cir.return

// LLVM:define dso_local void @clip_store_exception2
// LLVM:  [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM:  [[MEMBER:%.*]] = getelementptr %struct.clip, ptr [[LOAD]], i32 0, i32 0
// LLVM:  store volatile i24 3, ptr [[MEMBER]], align 4
// LLVM:  ret void

// OGCG: define dso_local void @clip_store_exception2
// OGCG:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// OGCG:   store volatile i24 3, ptr [[LOAD]], align 4
// OGCG:   ret void

void check_store_second_member (st4 *s4) {
  s4->b = 1;
}

// CIR:  cir.func dso_local @check_store_second_member
// CIR:    [[ONE:%.*]] = cir.const #cir.int<1> : !s32i
// CIR:    [[CAST:%.*]] = cir.cast integral [[ONE]] : !s32i -> !u64i
// CIR:    [[LOAD:%.*]] = cir.load align(8) {{.*}} : !cir.ptr<!cir.ptr<!rec_st4>>, !cir.ptr<!rec_st4>
// CIR:    [[MEMBER:%.*]] = cir.get_member [[LOAD]][2] {name = "b"} : !cir.ptr<!rec_st4> -> !cir.ptr<!u16i>
// CIR:    cir.set_bitfield align(8) (#bfi_b2, [[MEMBER]] : !cir.ptr<!u16i>, [[CAST]] : !u64i) {is_volatile} -> !u64i

// LLVM: define dso_local void @check_store_second_member
// LLVM:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// LLVM:   [[MEMBER:%.*]] = getelementptr %struct.st4, ptr [[LOAD]], i32 0, i32 2
// LLVM:   [[VAL:%.*]] = load volatile i64, ptr [[MEMBER]], align 8
// LLVM:   [[CLEAR:%.*]] = and i64 [[VAL]], -65536
// LLVM:   [[SET:%.*]] = or i64 [[CLEAR]], 1
// LLVM:   store volatile i64 [[SET]], ptr [[MEMBER]], align 8

// OGCG: define dso_local void @check_store_second_member
// OGCG:   [[LOAD:%.*]] = load ptr, ptr {{.*}}, align 8
// OGCG:   [[MEMBER:%.*]] = getelementptr inbounds i64, ptr [[LOAD]], i64 1
// OGCG:   [[LOADBF:%.*]] = load volatile i64, ptr [[MEMBER]], align 8
// OGCG:   [[CLR:%.*]] = and i64 [[LOADBF]], -65536
// OGCG:   [[SET:%.*]] = or i64 [[CLR]], 1
// OGCG:   store volatile i64 [[SET]], ptr [[MEMBER]], align 8
