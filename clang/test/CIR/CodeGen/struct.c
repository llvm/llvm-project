// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// For LLVM IR checks, the structs are defined before the variables, so these
// checks are at the top.
// CIR-DAG: !rec_IncompleteS = !cir.record<struct "IncompleteS" incomplete>
// CIR-DAG: !rec_CompleteS = !cir.record<struct "CompleteS" {!s32i, !s8i}>
// CIR-DAG: !rec_OuterS = !cir.record<struct "OuterS" {!rec_InnerS, !s32i}>  
// CIR-DAG: !rec_InnerS = !cir.record<struct "InnerS" {!s32i, !s8i}>
// CIR-DAG: !rec_PackedS = !cir.record<struct "PackedS" packed {!s32i, !s8i}>
// CIR-DAG: !rec_PackedAndPaddedS = !cir.record<struct "PackedAndPaddedS" packed padded {!s32i, !s8i, !u8i}>
// CIR-DAG: !rec_NodeS = !cir.record<struct "NodeS" {!cir.ptr<!cir.record<struct "NodeS">>}>
// CIR-DAG: !rec_RightS = !cir.record<struct "RightS" {!cir.ptr<!cir.record<struct "LeftS" {!cir.ptr<!cir.record<struct "RightS">>}>>}>
// CIR-DAG: !rec_LeftS = !cir.record<struct "LeftS" {!cir.ptr<!rec_RightS>}>
// CIR-DAG: !rec_CycleEnd = !cir.record<struct "CycleEnd" {!cir.ptr<!cir.record<struct "CycleStart" {!cir.ptr<!cir.record<struct "CycleMiddle" {!cir.ptr<!cir.record<struct "CycleEnd">>}>>}>>}>
// CIR-DAG: !rec_CycleMiddle = !cir.record<struct "CycleMiddle" {!cir.ptr<!rec_CycleEnd>}>
// CIR-DAG: !rec_CycleStart = !cir.record<struct "CycleStart" {!cir.ptr<!rec_CycleMiddle>}>
// LLVM-DAG: %struct.CompleteS = type { i32, i8 }
// LLVM-DAG: %struct.OuterS = type { %struct.InnerS, i32 }
// LLVM-DAG: %struct.InnerS = type { i32, i8 }
// LLVM-DAG: %struct.PackedS = type <{ i32, i8 }>
// LLVM-DAG: %struct.PackedAndPaddedS = type <{ i32, i8, i8 }>
// LLVM-DAG: %struct.NodeS = type { ptr }
// LLVM-DAG: %struct.LeftS = type { ptr }
// LLVM-DAG: %struct.RightS = type { ptr }
// LLVM-DAG: %struct.CycleStart = type { ptr }
// LLVM-DAG: %struct.CycleMiddle = type { ptr }
// LLVM-DAG: %struct.CycleEnd = type { ptr }
// OGCG-DAG: %struct.CompleteS = type { i32, i8 }
// OGCG-DAG: %struct.OuterS = type { %struct.InnerS, i32 }
// OGCG-DAG: %struct.InnerS = type { i32, i8 }
// OGCG-DAG: %struct.PackedS = type <{ i32, i8 }>
// OGCG-DAG: %struct.PackedAndPaddedS = type <{ i32, i8, i8 }>
// OGCG-DAG: %struct.NodeS = type { ptr }
// OGCG-DAG: %struct.LeftS = type { ptr }
// OGCG-DAG: %struct.RightS = type { ptr }
// OGCG-DAG: %struct.CycleStart = type { ptr }
// OGCG-DAG: %struct.CycleMiddle = type { ptr }
// OGCG-DAG: %struct.CycleEnd = type { ptr }

struct IncompleteS *p;

// CIR:      cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteS>
// LLVM-DAG: @p = dso_local global ptr null
// OGCG-DAG: @p = global ptr null, align 8

struct CompleteS {
  int a;
  char b;
} cs;

// CIR:       cir.global external @cs = #cir.zero : !rec_CompleteS
// LLVM-DAG:  @cs = dso_local global %struct.CompleteS zeroinitializer
// OGCG-DAG:  @cs = global %struct.CompleteS zeroinitializer, align 4

struct InnerS {
  int a;
  char b;
};

struct OuterS {
  struct InnerS is;
  int c;
};

struct OuterS os;

// CIR:       cir.global external @os = #cir.zero : !rec_OuterS
// LLVM-DAG:  @os = dso_local global %struct.OuterS zeroinitializer
// OGCG-DAG:  @os = global %struct.OuterS zeroinitializer, align 4

#pragma pack(push)
#pragma pack(1)

struct PackedS {
  int  a0;
  char a1;
} ps;

// CIR:       cir.global external @ps = #cir.zero : !rec_PackedS
// LLVM-DAG:  @ps = dso_local global %struct.PackedS zeroinitializer
// OGCG-DAG:  @ps = global %struct.PackedS zeroinitializer, align 1

struct PackedAndPaddedS {
  int  b0;
  char b1;
} __attribute__((aligned(2))) pps;

// CIR:       cir.global external @pps = #cir.zero : !rec_PackedAndPaddedS
// LLVM-DAG:  @pps = dso_local global %struct.PackedAndPaddedS zeroinitializer
// OGCG-DAG:  @pps = global %struct.PackedAndPaddedS zeroinitializer, align 2

#pragma pack(pop)

// Recursive type
struct NodeS {
  struct NodeS* next;
} node;

// CIR:      cir.global{{.*}} @node = #cir.zero : !rec_NodeS
// LLVM-DAG:  @node = dso_local global %struct.NodeS zeroinitializer
// OGCG-DAG:  @node = global %struct.NodeS zeroinitializer

// Mutually dependent types
struct RightS;
struct LeftS {
  struct RightS* right;
} ls;

// CIR:      cir.global{{.*}} @ls = #cir.zero : !rec_LeftS
// LLVM-DAG:  @ls = dso_local global %struct.LeftS zeroinitializer
// OGCG-DAG:  @ls = global %struct.LeftS zeroinitializer

struct RightS {
  struct LeftS* left;
} rs;

// CIR:      cir.global{{.*}} @rs = #cir.zero : !rec_RightS
// LLVM-DAG:  @rs = dso_local global %struct.RightS zeroinitializer
// OGCG-DAG:  @rs = global %struct.RightS zeroinitializer

struct CycleMiddle;
struct CycleEnd;
struct CycleStart {
  struct CycleMiddle* middle;
} start;

// CIR:      cir.global{{.*}} @start = #cir.zero : !rec_CycleStart
// LLVM-DAG:  @start = dso_local global %struct.CycleStart zeroinitializer
// OGCG-DAG:  @start = global %struct.CycleStart zeroinitializer

struct CycleMiddle {
  struct CycleEnd* end;
} middle;

// CIR:      cir.global{{.*}} @middle = #cir.zero : !rec_CycleMiddle
// LLVM-DAG:  @middle = dso_local global %struct.CycleMiddle zeroinitializer
// OGCG-DAG:  @middle = global %struct.CycleMiddle zeroinitializer

struct CycleEnd {
  struct CycleStart* start;
} end;

// CIR:      cir.global{{.*}} @end = #cir.zero : !rec_CycleEnd
// LLVM-DAG:  @end = dso_local global %struct.CycleEnd zeroinitializer
// OGCG-DAG:  @end = global %struct.CycleEnd zeroinitializer

void f(void) {
  struct IncompleteS *p;
}

// CIR:      cir.func @f()
// CIR-NEXT:   cir.alloca !cir.ptr<!rec_IncompleteS>, !cir.ptr<!cir.ptr<!rec_IncompleteS>>, ["p"] {alignment = 8 : i64}
// CIR-NEXT:   cir.return

// LLVM:      define void @f()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void

void f2(void) {
  struct CompleteS s;
}

// CIR:      cir.func @f2()
// CIR-NEXT:   cir.alloca !rec_CompleteS, !cir.ptr<!rec_CompleteS>, ["s"] {alignment = 4 : i64}
// CIR-NEXT:   cir.return

// LLVM:      define void @f2()
// LLVM-NEXT:   %[[S:.*]] = alloca %struct.CompleteS, i64 1, align 4
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @f2()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[S:.*]] = alloca %struct.CompleteS, align 4
// OGCG-NEXT:   ret void

char f3(int a) {
  cs.a = a;
  return cs.b;
}

// CIR:      cir.func @f3(%[[ARG_A:.*]]: !s32i
// CIR-NEXT:   %[[A_ADDR:.*]] = cir.alloca {{.*}} ["a", init] {alignment = 4 : i64}
// CIR-NEXT:   %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"] {alignment = 1 : i64}
// CIR-NEXT:   cir.store{{.*}} %[[ARG_A]], %[[A_ADDR]]
// CIR-NEXT:   %[[A_VAL:.*]] = cir.load{{.*}} %[[A_ADDR]]
// CIR-NEXT:   %[[CS:.*]] = cir.get_global @cs
// CIR-NEXT:   %[[CS_A:.*]] = cir.get_member %[[CS]][0] {name = "a"}
// CIR-NEXT:   cir.store{{.*}} %[[A_VAL]], %[[CS_A]]
// CIR-NEXT:   %[[CS2:.*]] = cir.get_global @cs
// CIR-NEXT:   %[[CS_B:.*]] = cir.get_member %[[CS2]][1] {name = "b"}
// CIR-NEXT:   %[[CS_B_VAL:.*]] = cir.load{{.*}} %[[CS_B]]
// CIR-NEXT:   cir.store{{.*}} %[[CS_B_VAL]], %[[RETVAL_ADDR]]
// CIR-NEXT:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR-NEXT:   cir.return %[[RETVAL]]

// LLVM:      define i8 @f3(i32 %[[ARG_A:.*]])
// LLVM-NEXT:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   %[[RETVAL_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM-NEXT:   store i32 %[[ARG_A]], ptr %[[A_ADDR]], align 4
// LLVM-NEXT:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM-NEXT:   store i32 %[[A_VAL]], ptr @cs, align 4
// LLVM-NEXT:   %[[CS_B_VAL:.*]] = load i8, ptr getelementptr inbounds nuw (i8, ptr @cs, i64 4), align 4
// LLVM-NEXT:   store i8 %[[CS_B_VAL]], ptr %[[RETVAL_ADDR]], align 1
// LLVM-NEXT:   %[[RETVAL:.*]] = load i8, ptr %[[RETVAL_ADDR]], align 1
// LLVM-NEXT:   ret i8 %[[RETVAL]]

// OGCG:      define{{.*}} i8 @f3(i32{{.*}} %[[ARG_A:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG-NEXT:   store i32 %[[ARG_A]], ptr %[[A_ADDR]], align 4
// OGCG-NEXT:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG-NEXT:   store i32 %[[A_VAL]], ptr @cs, align 4
// OGCG-NEXT:   %[[CS_B_VAL:.*]] = load i8, ptr getelementptr inbounds nuw (%struct.CompleteS, ptr @cs, i32 0, i32 1), align 4
// OGCG-NEXT:   ret i8 %[[CS_B_VAL]]

char f4(int a, struct CompleteS *p) {
  p->a = a;
  return p->b;
}

// CIR:      cir.func @f4(%[[ARG_A:.*]]: !s32i {{.*}}, %[[ARG_P:.*]]: !cir.ptr<!rec_CompleteS>
// CIR-NEXT:   %[[A_ADDR:.*]] = cir.alloca {{.*}} ["a", init] {alignment = 4 : i64}
// CIR-NEXT:   %[[P_ADDR:.*]] = cir.alloca {{.*}} ["p", init] {alignment = 8 : i64}
// CIR-NEXT:   %[[RETVAL_ADDR:.*]] = cir.alloca {{.*}} ["__retval"] {alignment = 1 : i64}
// CIR-NEXT:   cir.store{{.*}} %[[ARG_A]], %[[A_ADDR]]
// CIR-NEXT:   cir.store{{.*}} %[[ARG_P]], %[[P_ADDR]]
// CIR-NEXT:   %[[A_VAL:.*]] = cir.load{{.*}} %[[A_ADDR]]
// CIR-NEXT:   %[[P:.*]] = cir.load{{.*}} %[[P_ADDR]]
// CIR-NEXT:   %[[P_A:.*]] = cir.get_member %[[P]][0] {name = "a"}
// CIR-NEXT:   cir.store{{.*}} %[[A_VAL]], %[[P_A]]
// CIR-NEXT:   %[[P2:.*]] = cir.load{{.*}} %[[P_ADDR]]
// CIR-NEXT:   %[[P_B:.*]] = cir.get_member %[[P2]][1] {name = "b"}
// CIR-NEXT:   %[[P_B_VAL:.*]] = cir.load{{.*}} %[[P_B]]
// CIR-NEXT:   cir.store{{.*}} %[[P_B_VAL]], %[[RETVAL_ADDR]]
// CIR-NEXT:   %[[RETVAL:.*]] = cir.load{{.*}} %[[RETVAL_ADDR]]
// CIR-NEXT:   cir.return %[[RETVAL]]

// LLVM:      define i8 @f4(i32 %[[ARG_A:.*]], ptr %[[ARG_P:.*]])
// LLVM-NEXT:   %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   %[[P_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   %[[RETVAL_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM-NEXT:   store i32 %[[ARG_A]], ptr %[[A_ADDR]], align 4
// LLVM-NEXT:   store ptr %[[ARG_P]], ptr %[[P_ADDR]], align 8
// LLVM-NEXT:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM-NEXT:   %[[P_VAL:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// LLVM-NEXT:   %[[P_A:.*]] = getelementptr %struct.CompleteS, ptr %[[P_VAL]], i32 0, i32 0
// LLVM-NEXT:   store i32 %[[A_VAL]], ptr %[[P_A]], align 4
// LLVM-NEXT:   %[[P_VAL2:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// LLVM-NEXT:   %[[P_B:.*]] = getelementptr %struct.CompleteS, ptr %[[P_VAL2]], i32 0, i32 1
// LLVM-NEXT:   %[[P_B_VAL:.*]] = load i8, ptr %[[P_B]], align 4
// LLVM-NEXT:   store i8 %[[P_B_VAL]], ptr %[[RETVAL_ADDR]], align 1
// LLVM-NEXT:   %[[RETVAL:.*]] = load i8, ptr %[[RETVAL_ADDR]], align 1
// LLVM-NEXT:   ret i8 %[[RETVAL]]

// OGCG:      define{{.*}} i8 @f4(i32{{.*}} %[[ARG_A:.*]], ptr{{.*}} %[[ARG_P:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG-NEXT:   %[[P_ADDR:.*]] = alloca ptr, align 8
// OGCG-NEXT:   store i32 %[[ARG_A]], ptr %[[A_ADDR]], align 4
// OGCG-NEXT:   store ptr %[[ARG_P]], ptr %[[P_ADDR]], align 8
// OGCG-NEXT:   %[[A_VAL:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG-NEXT:   %[[P:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// OGCG-NEXT:   %[[P_A:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[P]], i32 0, i32 0
// OGCG-NEXT:   store i32 %[[A_VAL]], ptr %[[P_A]], align 4
// OGCG-NEXT:   %[[P2:.*]] = load ptr, ptr %[[P_ADDR]], align 8
// OGCG-NEXT:   %[[P_B:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[P2]], i32 0, i32 1
// OGCG-NEXT:   %[[P_B_VAL:.*]] = load i8, ptr %[[P_B]], align 4
// OGCG-NEXT:   ret i8 %[[P_B_VAL]]

void f5(struct NodeS* a) {
  a->next = 0;
}

// CIR: cir.func @f5
// CIR:   %[[NEXT:.*]] = cir.get_member {{%.}}[0] {name = "next"} : !cir.ptr<!rec_NodeS> -> !cir.ptr<!cir.ptr<!rec_NodeS>>
// CIR:   cir.store {{.*}}, %[[NEXT]]

// LLVM: define{{.*}} void @f5
// LLVM:   %[[NEXT:.*]] = getelementptr %struct.NodeS, ptr %{{.*}}, i32 0, i32 0
// LLVM:   store ptr null, ptr %[[NEXT]]

// OGCG: define{{.*}} void @f5
// OGCG:   %[[NEXT:.*]] = getelementptr inbounds nuw %struct.NodeS, ptr %{{.*}}, i32 0, i32 0
// OGCG:   store ptr null, ptr %[[NEXT]]

void f6(struct CycleStart *start) {
  struct CycleMiddle *middle = start->middle;
  struct CycleEnd *end = middle->end;
  struct CycleStart *start2 = end->start;
}

// CIR: cir.func @f6
// CIR:   %[[MIDDLE:.*]] = cir.get_member {{.*}}[0] {name = "middle"} : !cir.ptr<!rec_CycleStart> -> !cir.ptr<!cir.ptr<!rec_CycleMiddle>>
// CIR:   %[[END:.*]] = cir.get_member %{{.*}}[0] {name = "end"} : !cir.ptr<!rec_CycleMiddle> -> !cir.ptr<!cir.ptr<!rec_CycleEnd>>
// CIR:   %[[START2:.*]] = cir.get_member %{{.*}}[0] {name = "start"} : !cir.ptr<!rec_CycleEnd> -> !cir.ptr<!cir.ptr<!rec_CycleStart>>

// LLVM: define{{.*}} void @f6
// LLVM:   %[[MIDDLE:.*]] = getelementptr %struct.CycleStart, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %[[END:.*]] = getelementptr %struct.CycleMiddle, ptr %{{.*}}, i32 0, i32 0
// LLVM:   %[[START2:.*]] = getelementptr %struct.CycleEnd, ptr %{{.*}}, i32 0, i32 0

// OGCG: define{{.*}} void @f6
// OGCG:   %[[MIDDLE:.*]] = getelementptr inbounds nuw %struct.CycleStart, ptr %{{.*}}, i32 0, i32 0
// OGCG:   %[[END:.*]] = getelementptr inbounds nuw %struct.CycleMiddle, ptr %{{.*}}, i32 0, i32 0
// OGCG:   %[[START2:.*]] = getelementptr inbounds nuw %struct.CycleEnd, ptr %{{.*}}, i32 0, i32 0
