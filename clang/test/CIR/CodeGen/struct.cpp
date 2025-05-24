// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

struct IncompleteS;
IncompleteS *p;

// CIR: cir.global external @p = #cir.ptr<null> : !cir.ptr<!rec_IncompleteS>
// LLVM: @p = dso_local global ptr null
// OGCG: @p = global ptr null, align 8

struct CompleteS {
  int a;
  char b;
};

CompleteS cs;

// CIR:       cir.global external @cs = #cir.zero : !rec_CompleteS
// LLVM-DAG:  @cs = dso_local global %struct.CompleteS zeroinitializer
// OGCG-DAG:  @cs = global %struct.CompleteS zeroinitializer, align 4

void f(void) {
  IncompleteS *p;
}

// CIR:      cir.func @_Z1fv()
// CIR-NEXT:   cir.alloca !cir.ptr<!rec_IncompleteS>, !cir.ptr<!cir.ptr<!rec_IncompleteS>>, ["p"]
// CIR-NEXT:   cir.return

// LLVM:      define void @_Z1fv()
// LLVM-NEXT:   %[[P:.*]] = alloca ptr, i64 1, align 8
// LLVM-NEXT:   ret void

// OGCG:      define{{.*}} void @_Z1fv()
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[P:.*]] = alloca ptr, align 8
// OGCG-NEXT:   ret void

char f2(CompleteS &s) {
  return s.b;
}

// CIR: cir.func @_Z2f2R9CompleteS(%[[ARG_S:.*]]: !cir.ptr<!rec_CompleteS>{{.*}})
// CIR:   %[[S_ADDR:.*]] = cir.alloca !cir.ptr<!rec_CompleteS>, !cir.ptr<!cir.ptr<!rec_CompleteS>>, ["s", init, const]
// CIR:   cir.store %[[ARG_S]], %[[S_ADDR]]
// CIR:   %[[S_REF:.*]] = cir.load{{.*}} %[[S_ADDR]]
// CIR:   %[[S_ADDR2:.*]] = cir.get_member %[[S_REF]][1] {name = "b"}
// CIR:   %[[S_B:.*]] = cir.load{{.*}} %[[S_ADDR2]]

// LLVM: define i8 @_Z2f2R9CompleteS(ptr %[[ARG_S:.*]])
// LLVM:   %[[S_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG_S]], ptr %[[S_ADDR]]
// LLVM:   %[[S_REF:.*]] = load ptr, ptr %[[S_ADDR]], align 8
// LLVM:   %[[S_ADDR2:.*]] = getelementptr %struct.CompleteS, ptr %[[S_REF]], i32 0, i32 1
// LLVM:   %[[S_B:.*]] = load i8, ptr %[[S_ADDR2]]

// OGCG: define{{.*}} i8 @_Z2f2R9CompleteS(ptr{{.*}} %[[ARG_S:.*]])
// OGCG: entry:
// OGCG:   %[[S_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG_S]], ptr %[[S_ADDR]]
// OGCG:   %[[S_REF:.*]] = load ptr, ptr %[[S_ADDR]]
// OGCG:   %[[S_ADDR2:.*]] = getelementptr inbounds nuw %struct.CompleteS, ptr %[[S_REF]], i32 0, i32 1
// OGCG:   %[[S_B:.*]] = load i8, ptr %[[S_ADDR2]]
