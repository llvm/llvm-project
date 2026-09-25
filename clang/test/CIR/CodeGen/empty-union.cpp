// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,OGCG

// Padding-only union: CIR has no storage member and stores size in padding
// field, so getUnionStorageType() is null and getABIAlignment returns 1.
union Empty {};
// CIR-DAG: !rec_Empty = !cir.union<"Empty" {}, padding = {!u8i}>
// LLVM-DAG: %union.Empty = type { i8 }

// Aligned empty union (should have aligned integer member in CIR)
union alignas(16) EmptyAligned {};
// CIR-DAG: !rec_EmptyAligned = !cir.union<"EmptyAligned" {}, padding = {!cir.array<!u8i x 16>}>
// LLVM-DAG: %union.EmptyAligned = type { [16 x i8] }

// A zero-length bitfield is dropped during lowering, so this union reaches the
// same no-storage state despite declaring a member.
union OnlyZeroBitfield {
  int : 0;
};
// CIR-DAG: !rec_OnlyZeroBitfield = !cir.union<"OnlyZeroBitfield" {}, padding = {!u8i}>
// LLVM-DAG: %union.OnlyZeroBitfield = type { i8 }

// Struct holding a padding-only union member: layout queries !rec_Empty
// alignment (null largest), not OuterWithEmpty's int x.
union OuterWithEmpty {
  Empty e;
  int x;
};
struct WrapEmpty {
  OuterWithEmpty o;
  int s;
};
WrapEmpty w;
// CIR-DAG: !rec_OuterWithEmpty = !cir.union<"OuterWithEmpty" {data !rec_Empty, data !s32i}>
// CIR-DAG: !rec_WrapEmpty = !cir.struct<"WrapEmpty" {data !rec_OuterWithEmpty, data !s32i}>
// CIR-DAG: cir.global external @w = #cir.zero : !rec_WrapEmpty {alignment = 4 : i64}
// LLVM-DAG: %struct.WrapEmpty = type { %union.OuterWithEmpty, i32 }
// LLVM-DAG: %union.OuterWithEmpty = type { i32 }
// LLVM-DAG: @w = global %struct.WrapEmpty zeroinitializer, align 4

// A union whose only member is itself storage-less.  This one HAS a storage
// type, so it is the storage member's reported size that must be right.
union OnlyEmpty {
  Empty e;
};
// CIR-DAG: !rec_OnlyEmpty = !cir.union<"OnlyEmpty" {data !rec_Empty}>
// LLVM-DAG: %union.OnlyEmpty = type { %union.Empty }

// A storage-less union still occupies its own bytes inside a record, so the
// fields after it must not be pushed past them.
struct Leading {
  Empty e;
  int x;
};

struct Trailing {
  int x;
  Empty e;
};

struct Middle {
  int a;
  Empty e;
  int b;
};

struct LeadingOver {
  EmptyAligned e;
  int x;
};

struct LeadingZeroBitfield {
  OnlyZeroBitfield e;
  int x;
};

OnlyEmpty onlyEmpty;
Leading lead;
Trailing trail;
Middle mid;
LeadingOver leadOver;
LeadingZeroBitfield leadZero;
Leading leadArr[2];

// CIR-DAG: !rec_Leading = !cir.struct<"Leading" {data !rec_Empty, data !s32i}>
// CIR-DAG: !rec_Trailing = !cir.struct<"Trailing" {data !s32i, data !rec_Empty}>
// CIR-DAG: !rec_Middle = !cir.struct<"Middle" {data !s32i, data !rec_Empty, data !s32i}>
// CIR-DAG: !rec_LeadingOver = !cir.struct<"LeadingOver" {data !rec_EmptyAligned, data !s32i, pad !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_LeadingZeroBitfield = !cir.struct<"LeadingZeroBitfield" {data !rec_OnlyZeroBitfield, data !s32i}>

// CIR keeps the union's own named type as the record's field and leaves the
// bytes after it to the LLVM struct layout.  Classic covers the union together
// with those bytes in one char array.
// LLVMCIR-DAG: %struct.Leading = type { %union.Empty, i32 }
// LLVMCIR-DAG: %struct.Trailing = type { i32, %union.Empty }
// LLVMCIR-DAG: %struct.Middle = type { i32, %union.Empty, i32 }
// LLVMCIR-DAG: %struct.LeadingZeroBitfield = type { %union.OnlyZeroBitfield, i32 }
// LLVMCIR-DAG: %struct.LeadingOver = type { %union.EmptyAligned, i32, [12 x i8] }
// OGCG-DAG:    %struct.Leading = type { [4 x i8], i32 }
// OGCG-DAG:    %struct.Trailing = type { i32, [4 x i8] }
// OGCG-DAG:    %struct.Middle = type { i32, [4 x i8], i32 }
// OGCG-DAG:    %struct.LeadingZeroBitfield = type { [4 x i8], i32 }
// OGCG-DAG:    %struct.LeadingOver = type { [16 x i8], i32, [12 x i8] }
// LLVM-DAG:    @lead = global %struct.Leading zeroinitializer, align 4
// LLVM-DAG:    @leadOver = global %struct.LeadingOver zeroinitializer, align 16

void useEmpty() {
  Empty e;
}
// CIR: cir.func {{.*}}@_Z8useEmptyv()
// CIR:   cir.alloca "e" align(1) : !cir.ptr<!rec_Empty>
// LLVM: define {{.*}} void @_Z8useEmptyv()
// LLVM:   alloca %union.Empty{{.*}}, align 1

void useEmptyAligned() {
  EmptyAligned e;
}
// CIR: cir.func {{.*}}@_Z15useEmptyAlignedv()
// CIR:   cir.alloca "e" align(16) : !cir.ptr<!rec_EmptyAligned>
// LLVM: define {{.*}} void @_Z15useEmptyAlignedv()
// LLVM:   alloca %union.EmptyAligned{{.*}}, align 16

// Classic never refers to this union's type from inside LeadingZeroBitfield, so
// a variable of the type is what puts it in both modules.
void useZeroBitfield() {
  OnlyZeroBitfield e;
}
// CIR: cir.func {{.*}}@_Z15useZeroBitfieldv()
// CIR:   cir.alloca "e" align(1) : !cir.ptr<!rec_OnlyZeroBitfield>
// LLVM: define {{.*}} void @_Z15useZeroBitfieldv()
// LLVM:   alloca %union.OnlyZeroBitfield{{.*}}, align 1

// The union occupies one byte, so the int follows at offset 4.
int getLeading() { return lead.x; }

// CIR:  cir.func{{.*}} @_Z10getLeadingv()
// CIR:    %[[L:.*]] = cir.get_global @lead : !cir.ptr<!rec_Leading>
// CIR:    %{{.*}} = cir.get_member %[[L]][1] {name = "x"} : !cir.ptr<!rec_Leading> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z10getLeadingv()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @lead, i64 4), align 4

// With the union last, `x` stays at offset 0 whatever the union measures.  The
// record's own type above is what pins the tail.
int getTrailing() { return trail.x; }

// CIR:  cir.func{{.*}} @_Z11getTrailingv()
// CIR:    %[[T:.*]] = cir.get_global @trail : !cir.ptr<!rec_Trailing>
// CIR:    %{{.*}} = cir.get_member %[[T]][0] {name = "x"} : !cir.ptr<!rec_Trailing> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z11getTrailingv()
// LLVM:   load i32, ptr @trail, align 4

// The union sits between two fields, so only the field after it moves.
int getMiddle() { return mid.b; }

// CIR:  cir.func{{.*}} @_Z9getMiddlev()
// CIR:    %[[M:.*]] = cir.get_global @mid : !cir.ptr<!rec_Middle>
// CIR:    %{{.*}} = cir.get_member %[[M]][2] {name = "b"} : !cir.ptr<!rec_Middle> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z9getMiddlev()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @mid, i64 8), align 4

// An over-aligned union spells its size as an array of char rather than a
// single char, and the record embedding it has real tail padding of its own.
int getLeadingOver() { return leadOver.x; }

// CIR:  cir.func{{.*}} @_Z14getLeadingOverv()
// CIR:    %[[O:.*]] = cir.get_global @leadOver : !cir.ptr<!rec_LeadingOver>
// CIR:    %{{.*}} = cir.get_member %[[O]][1] {name = "x"} : !cir.ptr<!rec_LeadingOver> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z14getLeadingOverv()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @leadOver, i64 16), align 16

// The dropped bitfield leaves no storage member, so this behaves like Leading.
int getLeadingZeroBitfield() { return leadZero.x; }

// CIR:  cir.func{{.*}} @_Z22getLeadingZeroBitfieldv()
// CIR:    %[[Z:.*]] = cir.get_global @leadZero : !cir.ptr<!rec_LeadingZeroBitfield>
// CIR:    %{{.*}} = cir.get_member %[[Z]][1] {name = "x"} : !cir.ptr<!rec_LeadingZeroBitfield> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z22getLeadingZeroBitfieldv()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @leadZero, i64 4), align 4

// The element stride is 8, so the second element's int is at offset 12.
int getArray() { return leadArr[1].x; }

// CIR:  cir.func{{.*}} @_Z8getArrayv()
// CIR:    %[[A:.*]] = cir.get_global @leadArr : !cir.ptr<!cir.array<!rec_Leading x 2>>
// CIR:    %[[E:.*]] = cir.get_element %[[A]][%{{.*}} : !s64i] : !cir.ptr<!cir.array<!rec_Leading x 2>> -> !cir.ptr<!rec_Leading>
// CIR:    %{{.*}} = cir.get_member %[[E]][1] {name = "x"} : !cir.ptr<!rec_Leading> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z8getArrayv()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @leadArr, i64 12), align 4
