// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

class A {
    int a;
};

class B {
    int b;
public:
    A *getAsA();
};

class X : public A, public B {
    int x;
};

X *castAtoX(A *a) {
  return static_cast<X*>(a);
}

// CIR: cir.func {{.*}} @_Z8castAtoXP1A(%[[ARG0:.*]]: !cir.ptr<!rec_A> {{.*}})
// CIR:   %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>, ["a", init]
// CIR:   cir.store %[[ARG0]], %[[A_ADDR]] : !cir.ptr<!rec_A>, !cir.ptr<!cir.ptr<!rec_A>>
// CIR:   %[[A:.*]] = cir.load{{.*}} %[[A_ADDR]] : !cir.ptr<!cir.ptr<!rec_A>>, !cir.ptr<!rec_A>
// CIR:   %[[X:.*]] = cir.derived_class_addr %[[A]] : !cir.ptr<!rec_A> [0] -> !cir.ptr<!rec_X>

// Note: Because the offset is 0, a null check is not needed.

// LLVM: define {{.*}} ptr @_Z8castAtoXP1A(ptr %[[ARG0:.*]])
// LLVM:   %[[A_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[A_ADDR]]
// LLVM:   %[[X:.*]] = load ptr, ptr %[[A_ADDR]]

// OGCG: define {{.*}} ptr @_Z8castAtoXP1A(ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[A_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[A_ADDR]]
// OGCG:   %[[X:.*]] = load ptr, ptr %[[A_ADDR]]

X *castBtoX(B *b) {
  return static_cast<X*>(b);
}

// CIR: cir.func {{.*}} @_Z8castBtoXP1B(%[[ARG0:.*]]: !cir.ptr<!rec_B> {{.*}})
// CIR:   %[[B_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["b", init]
// CIR:   cir.store %[[ARG0]], %[[B_ADDR]] : !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>
// CIR:   %[[B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:   %[[X:.*]] = cir.derived_class_addr %[[B]] : !cir.ptr<!rec_B> [4] -> !cir.ptr<!rec_X>

// LLVM: define {{.*}} ptr @_Z8castBtoXP1B(ptr %[[ARG0:.*]])
// LLVM:   %[[B_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM:   store ptr %[[ARG0]], ptr %[[B_ADDR]], align 8
// LLVM:   %[[B:.*]] = load ptr, ptr %[[B_ADDR]], align 8
// LLVM:   %[[IS_NULL:.*]] = icmp eq ptr %[[B]], null
// LLVM:   %[[B_NON_NULL:.*]] = getelementptr inbounds i8, ptr %[[B]], i32 -4
// LLVM:   %[[X:.*]] = select i1 %[[IS_NULL]], ptr %[[B]], ptr %[[B_NON_NULL]]

// OGCG: define {{.*}} ptr @_Z8castBtoXP1B(ptr {{.*}} %[[ARG0:.*]])
// OGCG: entry:
// OGCG:   %[[B_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[B_ADDR]]
// OGCG:   %[[B:.*]] = load ptr, ptr %[[B_ADDR]]
// OGCG:   %[[IS_NULL:.*]] = icmp eq ptr %[[B]], null
// OGCG:   br i1 %[[IS_NULL]], label %[[LABEL_NULL:.*]], label %[[LABEL_NOTNULL:.*]]
// OGCG: [[LABEL_NOTNULL]]:
// OGCG:   %[[B_NON_NULL:.*]] = getelementptr inbounds i8, ptr %[[B]], i64 -4
// OGCG:   br label %[[LABEL_END:.*]]
// OGCG: [[LABEL_NULL]]:
// OGCG:   br label %[[LABEL_END:.*]]
// OGCG: [[LABEL_END]]:
// OGCG:   %[[X:.*]] = phi ptr [ %[[B_NON_NULL]], %[[LABEL_NOTNULL]] ], [ null, %[[LABEL_NULL]] ]

X &castBReftoXRef(B &b) {
  return static_cast<X&>(b);
}

// CIR: cir.func {{.*}} @_Z14castBReftoXRefR1B(%[[ARG0:.*]]: !cir.ptr<!rec_B> {{.*}})
// CIR:   %[[B_ADDR:.*]] = cir.alloca !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>, ["b", init, const]
// CIR:   cir.store %[[ARG0]], %[[B_ADDR]] : !cir.ptr<!rec_B>, !cir.ptr<!cir.ptr<!rec_B>>
// CIR:   %[[B:.*]] = cir.load{{.*}} %[[B_ADDR]] : !cir.ptr<!cir.ptr<!rec_B>>, !cir.ptr<!rec_B>
// CIR:   %[[X:.*]] = cir.derived_class_addr %[[B]] : !cir.ptr<!rec_B> nonnull [4] -> !cir.ptr<!rec_X>

// LLVM: define {{.*}} ptr @_Z14castBReftoXRefR1B(ptr %[[ARG0:.*]])
// LLVM:   %[[B_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[ARG0]], ptr %[[B_ADDR]]
// LLVM:   %[[B:.*]] = load ptr, ptr %[[B_ADDR]]
// LLVM:   %[[X:.*]] = getelementptr inbounds i8, ptr %[[B]], i32 -4

// OGCG: define {{.*}} ptr @_Z14castBReftoXRefR1B(ptr {{.*}} %[[ARG0:.*]])
// OGCG:   %[[B_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG0]], ptr %[[B_ADDR]]
// OGCG:   %[[B:.*]] = load ptr, ptr %[[B_ADDR]]
// OGCG:   %[[X:.*]] = getelementptr inbounds i8, ptr %[[B]], i64 -4
