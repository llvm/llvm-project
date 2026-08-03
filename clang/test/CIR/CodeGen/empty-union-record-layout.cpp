// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,OGCG

union Memberless {};

union alignas(16) MemberlessOver {};

// A zero-length bitfield is dropped during lowering, so this union reaches the
// same no-storage state despite declaring a member.
union OnlyZeroBitfield {
  int : 0;
};

struct Leading {
  Memberless e;
  int x;
};

struct Trailing {
  int x;
  Memberless e;
};

// A union whose only member is itself storage-less.  This one HAS a storage
// type, so it is the storage member's reported size that must be right, and a
// wrapping record cannot expose the error because the trailing field is
// realigned regardless.
union OnlyMemberless {
  Memberless e;
};

struct Middle {
  int a;
  Memberless e;
  int b;
};

struct LeadingOver {
  MemberlessOver e;
  int x;
};

struct LeadingZeroBitfield {
  OnlyZeroBitfield e;
  int x;
};

OnlyMemberless onlyMemberless;
Leading lead;
Trailing trail;
Middle mid;
LeadingOver leadOver;
LeadingZeroBitfield leadZero;
Leading leadArr[2];

// CIR-DAG: !rec_Memberless = !cir.union<"Memberless" {}, padding = {!u8i}>
// CIR-DAG: !rec_MemberlessOver = !cir.union<"MemberlessOver" {}, padding = {!cir.array<!u8i x 16>}>
// CIR-DAG: !rec_OnlyMemberless = !cir.union<"OnlyMemberless" {!rec_Memberless}>
// CIR-DAG: !rec_Leading = !cir.struct<"Leading" {!rec_Memberless, !s32i}>
// CIR-DAG: !rec_Trailing = !cir.struct<"Trailing" {!s32i, !rec_Memberless}>
// CIR-DAG: !rec_Middle = !cir.struct<"Middle" {!s32i, !rec_Memberless, !s32i}>
// CIR-DAG: !rec_LeadingOver = !cir.struct<"LeadingOver" padded {!rec_MemberlessOver, !s32i, !cir.array<!u8i x 12>}>
// CIR-DAG: !rec_OnlyZeroBitfield = !cir.union<"OnlyZeroBitfield" {}, padding = {!u8i}>
// CIR-DAG: !rec_LeadingZeroBitfield = !cir.struct<"LeadingZeroBitfield" {!rec_OnlyZeroBitfield, !s32i}>

// Neither path carries a pad for the union's own bytes, though they spell those
// bytes differently.
// LLVMCIR-DAG: %struct.Leading = type { %union.Memberless, i32 }
// LLVMCIR-DAG: %struct.Trailing = type { i32, %union.Memberless }
// LLVMCIR-DAG: %struct.Middle = type { i32, %union.Memberless, i32 }
// LLVMCIR-DAG: %struct.LeadingZeroBitfield = type { %union.OnlyZeroBitfield, i32 }
// LLVMCIR-DAG: %struct.LeadingOver = type { %union.MemberlessOver, i32, [12 x i8] }
// OGCG-DAG:    %struct.Leading = type { [4 x i8], i32 }
// OGCG-DAG:    %struct.Trailing = type { i32, [4 x i8] }
// OGCG-DAG:    %struct.Middle = type { i32, [4 x i8], i32 }
// OGCG-DAG:    %struct.LeadingZeroBitfield = type { [4 x i8], i32 }
// OGCG-DAG:    %struct.LeadingOver = type { [16 x i8], i32, [12 x i8] }
// LLVM-DAG:    %union.OnlyMemberless = type { %union.Memberless }
// LLVM-DAG:    @lead = global %struct.Leading zeroinitializer, align 4
// LLVM-DAG:    @leadOver = global %struct.LeadingOver zeroinitializer, align 16

// The union occupies one byte, so the int follows at offset 4.
int getLeading() { return lead.x; }

// CIR:  cir.func{{.*}} @_Z10getLeadingv()
// CIR:    %[[L:.*]] = cir.get_global @lead : !cir.ptr<!rec_Leading>
// CIR:    %{{.*}} = cir.get_member %[[L]][1] {name = "x"} : !cir.ptr<!rec_Leading> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z10getLeadingv()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @lead, i64 4), align 4

// With the union last, the size it contributes lands in the record's tail.
int getTrailing() { return trail.x; }

// CIR:  cir.func{{.*}} @_Z11getTrailingv()
// CIR:    %[[T:.*]] = cir.get_global @trail : !cir.ptr<!rec_Trailing>
// CIR:    %{{.*}} = cir.get_member %[[T]][0] {name = "x"} : !cir.ptr<!rec_Trailing> -> !cir.ptr<!s32i>
// LLVM: define dso_local noundef i32 @_Z11getTrailingv()
// LLVM:   load i32, ptr @trail, align 4

// The union sits between two fields, so only the field AFTER it moves.
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
// LLVM: define dso_local noundef i32 @_Z8getArrayv()
// LLVM:   load i32, ptr getelementptr inbounds nuw (i8, ptr @leadArr, i64 12), align 4
