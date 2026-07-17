// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexceptions -fcxx-exceptions -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -fexceptions -fcxx-exceptions -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// Aggregate conditional operators are emitted as cir.if; the branch regions
// are terminated after creation, so a throw arm must end at cir.unreachable
// with no trailing dead block (checked with CIR-NEXT below).

struct Agg { int x; int y; };
void take(Agg a);

// Baseline: both arms emit into the destination slot and the regions are
// closed with implicit terminators.
void init_normal(bool c) {
  Agg a = c ? Agg{1, 2} : Agg{3, 4};
}

// CIR-LABEL: cir.func{{.*}} @_Z11init_normalb(
// CIR:   %[[C:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[A:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!rec_Agg>
// CIR:   %[[C_VAL:.*]] = cir.load{{.*}} %[[C]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   cir.if %[[C_VAL]] {
// CIR:     %[[X:.*]] = cir.get_member %[[A]][0] {name = "x"}
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     cir.store{{.*}} %[[ONE]], %[[X]]
// CIR:     %[[Y:.*]] = cir.get_member %[[A]][1] {name = "y"}
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:     cir.store{{.*}} %[[TWO]], %[[Y]]
// CIR-NEXT:   } else {
// CIR:     %[[X2:.*]] = cir.get_member %[[A]][0] {name = "x"}
// CIR:     %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:     cir.store{{.*}} %[[THREE]], %[[X2]]
// CIR:     %[[Y2:.*]] = cir.get_member %[[A]][1] {name = "y"}
// CIR:     %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR:     cir.store{{.*}} %[[FOUR]], %[[Y2]]
// CIR-NEXT:   }
// CIR:   cir.return

// LLVM-LABEL: define{{.*}} void @_Z11init_normalb(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM:   br label %[[END:.*]]
// LLVM: [[FALSE_BB]]:
// LLVM:   store i32 3, ptr %{{.*}}
// LLVM:   store i32 4, ptr %{{.*}}
// LLVM:   br label %[[END]]
// LLVM: [[END]]:
// LLVM:   ret void

// OGCG-LABEL: define{{.*}} void @_Z11init_normalb(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[FALSE_BB]]:
// OGCG:   store i32 3, ptr %{{.*}}
// OGCG:   store i32 4, ptr %{{.*}}
// OGCG:   br label %[[END]]
// OGCG: [[END]]:
// OGCG:   ret void

// Assignment context: the conditional materializes into a temporary that is
// then assigned to the target.
void assign_throw(bool c, Agg &a) {
  a = c ? throw 0 : Agg{1, 2};
}

// CIR-LABEL: cir.func{{.*}} @_Z12assign_throwbR3Agg(
// CIR:   %[[C:.*]] = cir.alloca "c" {{.*}} init : !cir.ptr<!cir.bool>
// CIR:   %[[A_REF:.*]] = cir.alloca "a" {{.*}} init const : !cir.ptr<!cir.ptr<!rec_Agg>>
// CIR:   %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_Agg>
// CIR:   %[[C_VAL:.*]] = cir.load{{.*}} %[[C]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR:   cir.if %[[C_VAL]] {
// CIR:     %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   } else {
// CIR:     cir.get_member %[[TMP]][0] {name = "x"}
// CIR:     cir.get_member %[[TMP]][1] {name = "y"}
// CIR:   }
// CIR:   %[[A_VAL:.*]] = cir.load %[[A_REF]]
// CIR:   cir.call @_ZN3AggaSEOS_(%[[A_VAL]], %[[TMP]])

// LLVM-LABEL: define{{.*}} void @_Z12assign_throwbR3Agg(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   call{{.*}} ptr @__cxa_allocate_exception
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM:   br label %[[END:.*]]
// LLVM: [[END]]:
// LLVM:   call{{.*}} ptr @_ZN3AggaSEOS_(

// OGCG-LABEL: define{{.*}} void @_Z12assign_throwbR3Agg(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   call{{.*}} ptr @__cxa_allocate_exception
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG:   br label %[[END:.*]]
// OGCG: [[END]]:
// OGCG:   call void @llvm.memcpy.p0.p0.i64(ptr align 4 %{{.*}}, ptr align 4 %{{.*}}, i64 8

// Nested conditional: the inner throw arm terminates the inner cir.if region
// directly.
void nested_throw(bool c1, bool c2) {
  Agg a = c1 ? (c2 ? throw 0 : Agg{1, 2}) : Agg{3, 4};
}

// CIR-LABEL: cir.func{{.*}} @_Z12nested_throwbb(
// CIR:   %[[A:.*]] = cir.alloca "a" {{.*}} init : !cir.ptr<!rec_Agg>
// CIR:   cir.if %{{.*}} {
// CIR:     %[[C2_VAL:.*]] = cir.load{{.*}} : !cir.ptr<!cir.bool>, !cir.bool
// CIR:     cir.if %[[C2_VAL]] {
// CIR:       %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:       cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:       cir.unreachable
// CIR-NEXT:     } else {
// CIR:       cir.get_member %[[A]][0] {name = "x"}
// CIR:       cir.get_member %[[A]][1] {name = "y"}
// CIR:     }
// CIR:   } else {
// CIR:     cir.get_member %[[A]][0] {name = "x"}
// CIR:     cir.get_member %[[A]][1] {name = "y"}
// CIR:   }
// CIR:   cir.return

// LLVM-LABEL: define{{.*}} void @_Z12nested_throwbb(
// LLVM:   br i1 %{{.*}}, label %[[OUTER_TRUE:.*]], label %[[OUTER_FALSE:.*]]
// LLVM: [[OUTER_TRUE]]:
// LLVM:   br i1 %{{.*}}, label %[[INNER_TRUE:.*]], label %[[INNER_FALSE:.*]]
// LLVM: [[INNER_TRUE]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[INNER_FALSE]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM: [[OUTER_FALSE]]:
// LLVM:   store i32 3, ptr %{{.*}}
// LLVM:   store i32 4, ptr %{{.*}}

// OGCG-LABEL: define{{.*}} void @_Z12nested_throwbb(
// OGCG:   br i1 %{{.*}}, label %[[OUTER_TRUE:.*]], label %[[OUTER_FALSE:.*]]
// OGCG: [[OUTER_TRUE]]:
// OGCG:   br i1 %{{.*}}, label %[[INNER_TRUE:.*]], label %[[INNER_FALSE:.*]]
// OGCG: [[INNER_TRUE]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[INNER_FALSE]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG: [[OUTER_FALSE]]:
// OGCG:   store i32 3, ptr %{{.*}}
// OGCG:   store i32 4, ptr %{{.*}}

// Call-argument context: the conditional materializes the argument temporary.
void arg_throw(bool c) {
  take(c ? throw 0 : Agg{1, 2});
}

// CIR-LABEL: cir.func{{.*}} @_Z9arg_throwb(
// CIR:   %[[TMP:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<!rec_Agg>
// CIR:   cir.if %{{.*}} {
// CIR:     %[[EXC:.*]] = cir.alloc.exception{{.*}} -> !cir.ptr<!s32i>
// CIR:     cir.throw %[[EXC]] : !cir.ptr<!s32i>, @_ZTIi
// CIR:     cir.unreachable
// CIR-NEXT:   } else {
// CIR:     cir.get_member %[[TMP]][0] {name = "x"}
// CIR:     cir.get_member %[[TMP]][1] {name = "y"}
// CIR:   }
// CIR:   %[[ARG:.*]] = cir.load{{.*}} %[[TMP]] : !cir.ptr<!rec_Agg>, !rec_Agg
// CIR:   cir.call @_Z4take3Agg(%[[ARG]])

// LLVM-LABEL: define{{.*}} void @_Z9arg_throwb(
// LLVM:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// LLVM: [[TRUE_BB]]:
// LLVM:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// LLVM:   unreachable
// LLVM: [[FALSE_BB]]:
// LLVM:   store i32 1, ptr %{{.*}}
// LLVM:   store i32 2, ptr %{{.*}}
// LLVM:   br label %[[END:.*]]
// LLVM: [[END]]:
// LLVM:   call void @_Z4take3Agg(

// OGCG-LABEL: define{{.*}} void @_Z9arg_throwb(
// OGCG:   br i1 %{{.*}}, label %[[TRUE_BB:.*]], label %[[FALSE_BB:.*]]
// OGCG: [[TRUE_BB]]:
// OGCG:   call void @__cxa_throw(ptr %{{.*}}, ptr @_ZTIi
// OGCG:   unreachable
// OGCG: [[FALSE_BB]]:
// OGCG:   store i32 1, ptr %{{.*}}
// OGCG:   store i32 2, ptr %{{.*}}
// OGCG: [[END:.*]]:
// OGCG:   call void @_Z4take3Agg(
