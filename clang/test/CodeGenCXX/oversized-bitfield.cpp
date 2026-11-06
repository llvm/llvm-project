// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fdump-record-layouts-simple -o %t.ll %s | FileCheck %s --check-prefix=LAYOUT
// RUN: FileCheck %s --check-prefix=IR < %t.ll
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -debug-info-kind=limited -o - %s | FileCheck %s --check-prefixes=IR,DEBUG

// An over-wide C++ bit-field only has type-width value bits. Widths of 2^15
// and 2^16 used to wrap to zero in CGBitFieldInfo::Size and crash (GH119207).

#pragma clang diagnostic ignored "-Wbitfield-width"

// IR: @g32769 = {{.*}}global { i8, i8, i8, i8, [4100 x i8] } { i8 120, i8 86, i8 52, i8 18, [4100 x i8] undef }

enum E {
  a = 1,
  b = 2,
  c = 3
};

struct S {
  E e1 : (1 << 16);
};

// LAYOUT-LABEL: LLVMType:%struct.S =
// LAYOUT-SAME: type { i65536 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:0 StorageSize:65536 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i32 @_Z4loadP1S(
// IR: [[LOAD:%.*]] = load i65536, ptr
// IR-NEXT: [[CLEAR:%.*]] = and i65536 [[LOAD]], 4294967295
// IR-NEXT: trunc i65536 [[CLEAR]] to i32
E load(S *s) {
  return s->e1;
}

struct foo
{
  int bar   : 32768;
};

// LAYOUT-LABEL: LLVMType:%struct.foo =
// LAYOUT-SAME: type { i32768 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:32768 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i32 @main(
// IR: alloca %struct.foo, align 8
int main() {
  foo x;
}

struct W32767 { int w32767 : 32767; };
struct W32768 { int w32768 : 32768; };
struct W32769 { int w32769 : 32769; };
struct W65537 { long long w65537 : 65537; };

// LAYOUT-LABEL: LLVMType:%struct.W32767 =
// LAYOUT-SAME: type { i32768 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:32768 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i32 @_Z7ld32767P6W32767(
// IR: [[LOAD:%.*]] = load i32768, ptr
// IR-NEXT: [[SHL:%.*]] = shl i32768 [[LOAD]], 32736
// IR-NEXT: [[ASHR:%.*]] = ashr i32768 [[SHL]], 32736
// IR-NEXT: trunc i32768 [[ASHR]] to i32
int ld32767(W32767 *p) { return p->w32767; }

// LAYOUT-LABEL: LLVMType:%struct.W32768 =
// LAYOUT-SAME: type { i32768 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:32768 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i32 @_Z7ld32768P6W32768(
// IR: [[LOAD:%.*]] = load i32768, ptr
// IR-NEXT: [[SHL:%.*]] = shl i32768 [[LOAD]], 32736
// IR-NEXT: [[ASHR:%.*]] = ashr i32768 [[SHL]], 32736
// IR-NEXT: trunc i32768 [[ASHR]] to i32
int ld32768(W32768 *p) { return p->w32768; }

// LAYOUT-LABEL: LLVMType:%struct.W32769 =
// LAYOUT-SAME: type { [4097 x i8], [7 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:32776 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i32 @_Z7ld32769P6W32769(
// IR: [[LOAD:%.*]] = load i32776, ptr
// IR-NEXT: [[SHL:%.*]] = shl i32776 [[LOAD]], 32744
// IR-NEXT: [[ASHR:%.*]] = ashr i32776 [[SHL]], 32744
// IR-NEXT: trunc i32776 [[ASHR]] to i32
int ld32769(W32769 *p) { return p->w32769; }

// IR-LABEL: define {{.*}} void @_Z7st32769P6W32769i(
// IR: [[EXT:%.*]] = zext i32 {{.*}} to i32776
// IR-NEXT: [[OLD:%.*]] = load i32776, ptr
// IR-NEXT: [[VAL:%.*]] = and i32776 [[EXT]], 4294967295
// IR-NEXT: [[KEPT:%.*]] = and i32776 [[OLD]], -4294967296
// IR-NEXT: [[SET:%.*]] = or i32776 [[KEPT]], [[VAL]]
// IR-NEXT: store i32776 [[SET]], ptr
void st32769(W32769 *p, int v) { p->w32769 = v; }

W32769 g32769 = {0x12345678};

// LAYOUT-LABEL: LLVMType:%struct.W65537 =
// LAYOUT-SAME: type { [8193 x i8], [7 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:64 IsSigned:1 StorageSize:65544 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i64 @_Z7ld65537P6W65537(
// IR: [[LOAD:%.*]] = load i65544, ptr
// IR-NEXT: [[SHL:%.*]] = shl i65544 [[LOAD]], 65480
// IR-NEXT: [[ASHR:%.*]] = ashr i65544 [[SHL]], 65480
// IR-NEXT: trunc i65544 [[ASHR]] to i64
long long ld65537(W65537 *p) { return p->w65537; }

union U {
  int w : 32768;
};

// LAYOUT-LABEL: LLVMType:%union.U =
// LAYOUT-SAME: type { i32, [4092 x i8] }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:32 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i32 @_Z3ldUP1U(
// IR: [[LOAD:%.*]] = load i32, ptr
// IR-NEXT: ret i32 [[LOAD]]
int ldU(U *p) { return p->w; }

struct Control {
  int lo : 7;
  unsigned hi : 25;
};

// LAYOUT-LABEL: LLVMType:%struct.Control =
// LAYOUT-SAME: type { i32 }
// LAYOUT: BitFields:[
// LAYOUT-NEXT: <CGBitFieldInfo Offset:0 Size:7 IsSigned:1 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: <CGBitFieldInfo Offset:7 Size:25 IsSigned:0 StorageSize:32 StorageOffset:0
// LAYOUT-NEXT: ]>

// IR-LABEL: define {{.*}} i32 @_Z5ldCtlP7Control(
// IR: [[LOAD:%.*]] = load i32, ptr
// IR-NEXT: [[SHL:%.*]] = shl i32 [[LOAD]], 25
// IR-NEXT: [[ASHR:%.*]] = ashr i32 [[SHL]], 25
// IR-NEXT: ret i32 [[ASHR]]
int ldCtl(Control *p) { return p->lo; }

// DEBUG-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "S", {{.*}}size: 65536,
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "e1", {{.*}}baseType: !{{[0-9]+}}, size: 32, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DICompositeType(tag: DW_TAG_structure_type, name: "foo", {{.*}}size: 32768,
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "bar", {{.*}}baseType: !{{[0-9]+}}, size: 32, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "w32767", {{.*}}size: 32, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "w32768", {{.*}}size: 32, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "w32769", {{.*}}size: 32, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "w65537", {{.*}}size: 64, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "w", {{.*}}size: 32, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "lo", {{.*}}size: 7, flags: DIFlagBitField, extraData: i64 0)
// DEBUG-DAG: !DIDerivedType(tag: DW_TAG_member, name: "hi", {{.*}}size: 25, offset: 7, flags: DIFlagBitField, extraData: i64 0)
