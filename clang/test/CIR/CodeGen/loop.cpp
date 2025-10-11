// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

void l0() {
  for (;;) {
  }
}

// CIR: cir.func{{.*}} @_Z2l0v
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.condition(%[[TRUE]])
// CIR:     } body {
// CIR:       cir.yield
// CIR:     } step {
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define{{.*}} void @_Z2l0v()
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l0v()
// OGCG: entry:
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_COND]]

void l1() {
  for (int i = 0; ; ) {
  }
}

// CIR:      cir.func{{.*}} @_Z2l1v
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:     %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:     cir.store{{.*}} %[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:     cir.for : cond {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } body {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } step {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: define{{.*}} void @_Z2l1v()
// LLVM:   %[[I:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   store i32 0, ptr %[[I]], align 4
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l1v()
// OGCG: entry:
// OGCG:   %[[I:.*]] = alloca i32, align 4
// OGCG:   store i32 0, ptr %[[I]], align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_COND]]

void l2() {
  for (;;) {
    int i = 0;
  }
}

// CIR:      cir.func{{.*}} @_Z2l2v
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     cir.for : cond {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } body {
// CIR-NEXT:       cir.scope {
// CIR-NEXT:         %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:         %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:         cir.store{{.*}} %[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:       }
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } step {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: define{{.*}} void @_Z2l2v()
// LLVM:   %[[I:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   store i32 0, ptr %[[I]], align 4
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l2v()
// OGCG: entry:
// OGCG:   %[[I:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   store i32 0, ptr %[[I]], align 4
// OGCG:   br label %[[FOR_COND]]

// This is the same as l2 but without a compound statement for the body.
void l3() {
  for (;;)
    int i = 0;
}

// CIR:      cir.func{{.*}} @_Z2l3v
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:     cir.for : cond {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } body {
// CIR-NEXT:       %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:       cir.store{{.*}} %[[ZERO]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } step {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     }
// CIR-NEXT:   }
// CIR-NEXT:   cir.return
// CIR-NEXT: }

// LLVM: define{{.*}} void @_Z2l3v()
// LLVM:   %[[I:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL5:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   store i32 0, ptr %[[I]], align 4
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL5]]:
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l3v()
// OGCG: entry:
// OGCG:   %[[I:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   store i32 0, ptr %[[I]], align 4
// OGCG:   br label %[[FOR_COND]]

void l4() {
  int a[10];
  for (int n : a)
    ;
}

// CIR: cir.func{{.*}} @_Z2l4v
// CIR:   %[[A_ADDR:.*]] = cir.alloca {{.*}} ["a"]
// CIR:   cir.scope {
// CIR:     %[[RANGE_ADDR:.*]] = cir.alloca {{.*}} ["__range1", init, const]
// CIR:     %[[BEGIN_ADDR:.*]] = cir.alloca {{.*}} ["__begin1", init]
// CIR:     %[[END_ADDR:.*]] = cir.alloca {{.*}} ["__end1", init]
// CIR:     %[[N_ADDR:.*]] = cir.alloca {{.*}} ["n", init]
// CIR:     cir.store{{.*}} %[[A_ADDR]], %[[RANGE_ADDR]]
// CIR:     %[[RANGE_LOAD:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:     %[[RANGE_CAST:.*]] = cir.cast array_to_ptrdecay %[[RANGE_LOAD]] : {{.*}}
// CIR:     cir.store{{.*}} %[[RANGE_CAST]], %[[BEGIN_ADDR]]
// CIR:     %[[BEGIN:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:     %[[BEGIN_CAST:.*]] = cir.cast array_to_ptrdecay %[[BEGIN]] : {{.*}}
// CIR:     %[[TEN:.*]] = cir.const #cir.int<10>
// CIR:     %[[END_PTR:.*]] = cir.ptr_stride %[[BEGIN_CAST]], %[[TEN]] : ({{.*}}, {{.*}})
// CIR:     cir.store{{.*}} %[[END_PTR]], %[[END_ADDR]]
// CIR:     cir.for : cond {
// CIR:       %[[CUR:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:       %[[END:.*]] = cir.load{{.*}} %[[END_ADDR]]
// CIR:       %[[CMP:.*]] = cir.cmp(ne, %[[CUR]], %[[END]])
// CIR:       cir.condition(%[[CMP]])
// CIR:     } body {
// CIR:       %[[CUR:.*]] = cir.load deref{{.*}} %[[BEGIN_ADDR]]
// CIR:       %[[N:.*]] = cir.load{{.*}} %[[CUR]]
// CIR:       cir.store{{.*}} %[[N]], %[[N_ADDR]]
// CIR:       cir.yield
// CIR:     } step {
// CIR:       %[[CUR:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:       %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[ONE]] : ({{.*}}, {{.*}})
// CIR:       cir.store{{.*}} %[[NEXT]], %[[BEGIN_ADDR]]
// CIR:       cir.yield
// CIR:     }
// CIR:   }

// LLVM: define{{.*}} void @_Z2l4v() {
// LLVM:   %[[RANGE_ADDR:.*]] = alloca ptr
// LLVM:   %[[BEGIN_ADDR:.*]] = alloca ptr
// LLVM:   %[[END_ADDR:.*]] = alloca ptr
// LLVM:   %[[N_ADDR:.*]] = alloca i32
// LLVM:   %[[A_ADDR:.*]] = alloca [10 x i32]
// LLVM:   br label %[[SETUP:.*]]
// LLVM: [[SETUP]]:
// LLVM:   store ptr %[[A_ADDR]], ptr %[[RANGE_ADDR]]
// LLVM:   %[[BEGIN:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// LLVM:   %[[BEGIN_CAST:.*]] = getelementptr i32, ptr %[[BEGIN]], i32 0
// LLVM:   store ptr %[[BEGIN_CAST]], ptr %[[BEGIN_ADDR]]
// LLVM:   %[[RANGE:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// LLVM:   %[[RANGE_CAST:.*]] = getelementptr i32, ptr %[[RANGE]], i32 0
// LLVM:   %[[END_PTR:.*]] = getelementptr i32, ptr %[[RANGE_CAST]], i64 10
// LLVM:   store ptr %[[END_PTR]], ptr %[[END_ADDR]]
// LLVM:   br label %[[COND:.*]]
// LLVM: [[COND]]:
// LLVM:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// LLVM:   %[[END:.*]] = load ptr, ptr %[[END_ADDR]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[BEGIN]], %[[END]]
// LLVM:   br i1 %[[CMP]], label %[[BODY:.*]], label %[[END:.*]]
// LLVM: [[BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// LLVM:   %[[A_CUR:.*]] = load i32, ptr %[[CUR]]
// LLVM:   store i32 %[[A_CUR]], ptr %[[N_ADDR]]
// LLVM:   br label %[[STEP:.*]]
// LLVM: [[STEP]]:
// LLVM:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// LLVM:   %[[NEXT:.*]] = getelementptr i32, ptr %[[BEGIN]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[BEGIN_ADDR]]
// LLVM:   br label %[[COND]]
// LLVM: [[END]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l4v()
// OGCG:   %[[A_ADDR:.*]] = alloca [10 x i32]
// OGCG:   %[[RANGE_ADDR:.*]] = alloca ptr
// OGCG:   %[[BEGIN_ADDR:.*]] = alloca ptr
// OGCG:   %[[END_ADDR:.*]] = alloca ptr
// OGCG:   %[[N_ADDR:.*]] = alloca i32
// OGCG:   store ptr %[[A_ADDR]], ptr %[[RANGE_ADDR]]
// OGCG:   %[[BEGIN:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// OGCG:   %[[BEGIN_CAST:.*]] = getelementptr inbounds [10 x i32], ptr %[[BEGIN]], i64 0, i64 0
// OGCG:   store ptr %[[BEGIN_CAST]], ptr %[[BEGIN_ADDR]]
// OGCG:   %[[RANGE:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// OGCG:   %[[RANGE_CAST:.*]] = getelementptr inbounds [10 x i32], ptr %[[RANGE]], i64 0, i64 0
// OGCG:   %[[END_PTR:.*]] = getelementptr inbounds i32, ptr %[[RANGE_CAST]], i64 10
// OGCG:   store ptr %[[END_PTR]], ptr %[[END_ADDR]]
// OGCG:   br label %[[COND:.*]]
// OGCG: [[COND]]:
// OGCG:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// OGCG:   %[[END:.*]] = load ptr, ptr %[[END_ADDR]]
// OGCG:   %[[CMP:.*]] = icmp ne ptr %[[BEGIN]], %[[END]]
// OGCG:   br i1 %[[CMP]], label %[[BODY:.*]], label %[[END:.*]]
// OGCG: [[BODY]]:
// OGCG:   %[[CUR:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// OGCG:   %[[A_CUR:.*]] = load i32, ptr %[[CUR]]
// OGCG:   store i32 %[[A_CUR]], ptr %[[N_ADDR]]
// OGCG:   br label %[[STEP:.*]]
// OGCG: [[STEP]]:
// OGCG:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// OGCG:   %[[NEXT:.*]] = getelementptr inbounds nuw i32, ptr %[[BEGIN]], i32 1
// OGCG:   store ptr %[[NEXT]], ptr %[[BEGIN_ADDR]]
// OGCG:   br label %[[COND]]
// OGCG: [[END]]:
// OGCG:   ret void

void l5() {
  for (int arr[]{1,2,3,4}; auto x : arr) {} 
}

// CIR: cir.func{{.*}} @_Z2l5v
// CIR:   cir.scope {
// CIR:     %[[ARR_ADDR:.*]] = cir.alloca {{.*}} ["arr", init]
// CIR:     %[[RANGE_ADDR:.*]] = cir.alloca {{.*}} ["__range1", init, const]
// CIR:     %[[BEGIN_ADDR:.*]] = cir.alloca {{.*}} ["__begin1", init]
// CIR:     %[[END_ADDR:.*]] = cir.alloca {{.*}} ["__end1", init]
// CIR:     %[[X_ADDR:.*]] = cir.alloca {{.*}} ["x", init]
// CIR:     %[[ARR_CAST:.*]] = cir.cast array_to_ptrdecay %[[ARR_ADDR]] : {{.*}}
// CIR:     %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:     cir.store{{.*}} %[[ONE]], %[[ARR_CAST]]
// CIR:     %[[OFFSET1:.*]] = cir.const #cir.int<1> : !s64i
// CIR:     %[[STRIDE:.*]] = cir.ptr_stride %[[ARR_CAST]], %[[OFFSET1]] : ({{.*}}, {{.*}})
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR:     cir.store{{.*}} %[[TWO]], %[[STRIDE]]
// CIR:     %[[OFFSET2:.*]] = cir.const #cir.int<2> : !s64i
// CIR:     %[[STRIDE2:.*]] = cir.ptr_stride %[[ARR_CAST]], %[[OFFSET2]] : ({{.*}}, {{.*}})
// CIR:     %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:     cir.store{{.*}} %[[THREE]], %[[STRIDE2]]
// CIR:     %[[OFFSET3:.*]] = cir.const #cir.int<3> : !s64i
// CIR:     %[[STRIDE3:.*]] = cir.ptr_stride %[[ARR_CAST]], %[[OFFSET3]] : ({{.*}}, {{.*}})
// CIR:     %[[FOUR:.*]] = cir.const #cir.int<4> : !s32i
// CIR:     cir.store{{.*}} %[[FOUR]], %[[STRIDE3]]
// CIR:     cir.store{{.*}} %[[ARR_ADDR]], %[[RANGE_ADDR]]
// CIR:     %[[RANGE_LOAD:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:     %[[RANGE_CAST:.*]] = cir.cast array_to_ptrdecay %[[RANGE_LOAD]] : {{.*}}
// CIR:     cir.store{{.*}} %[[RANGE_CAST]], %[[BEGIN_ADDR]]
// CIR:     %[[BEGIN:.*]] = cir.load{{.*}} %[[RANGE_ADDR]]
// CIR:     %[[BEGIN_CAST:.*]] = cir.cast array_to_ptrdecay %[[BEGIN]] : {{.*}}
// CIR:     %[[FOUR:.*]] = cir.const #cir.int<4> : !s64i
// CIR:     %[[END_PTR:.*]] = cir.ptr_stride %[[BEGIN_CAST]], %[[FOUR]] : ({{.*}}, {{.*}})
// CIR:     cir.store{{.*}} %[[END_PTR]], %[[END_ADDR]]
// CIR:     cir.for : cond {
// CIR:       %[[CUR:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:       %[[END:.*]] = cir.load{{.*}} %[[END_ADDR]]
// CIR:       %[[CMP:.*]] = cir.cmp(ne, %[[CUR]], %[[END]])
// CIR:       cir.condition(%[[CMP]])
// CIR:     } body {
// CIR:       %[[CUR:.*]] = cir.load deref{{.*}} %[[BEGIN_ADDR]]
// CIR:       %[[X:.*]] = cir.load{{.*}} %[[CUR]]
// CIR:       cir.store{{.*}} %[[X]], %[[X_ADDR]]
// CIR:       cir.yield
// CIR:     } step {
// CIR:       %[[CUR:.*]] = cir.load{{.*}} %[[BEGIN_ADDR]]
// CIR:       %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:       %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[ONE]] : ({{.*}}, {{.*}})
// CIR:       cir.store{{.*}} %[[NEXT]], %[[BEGIN_ADDR]]
// CIR:       cir.yield
// CIR:     }
// CIR:   }

// LLVM: define{{.*}} void @_Z2l5v() {
// LLVM:   %[[ARR_ADDR:.*]] = alloca [4 x i32]
// LLVM:   %[[RANGE_ADDR:.*]] = alloca ptr
// LLVM:   %[[BEGIN_ADDR:.*]] = alloca ptr
// LLVM:   %[[END_ADDR:.*]] = alloca ptr
// LLVM:   %[[X_ADDR:.*]] = alloca i32
// LLVM:   br label %[[SETUP:.*]]
// LLVM: [[SETUP]]:
// LLVM:   %[[ARR_0:.*]] = getelementptr i32, ptr %[[ARR_ADDR]], i32 0
// LLVM:   store i32 1, ptr %[[ARR_0]]
// LLVM:   %[[ARR_1:.*]] = getelementptr i32, ptr %[[ARR_0]], i64 1
// LLVM:   store i32 2, ptr %[[ARR_1]]
// LLVM:   %[[ARR_2:.*]] = getelementptr i32, ptr %[[ARR_0]], i64 2
// LLVM:   store i32 3, ptr %[[ARR_2]]
// LLVM:   %[[ARR_3:.*]] = getelementptr i32, ptr %[[ARR_0]], i64 3
// LLVM:   store i32 4, ptr %[[ARR_3]]
// LLVM:   store ptr %[[ARR_ADDR]], ptr %[[RANGE_ADDR]]
// LLVM:   %[[BEGIN:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// LLVM:   %[[BEGIN_CAST:.*]] = getelementptr i32, ptr %[[BEGIN]], i32 0
// LLVM:   store ptr %[[BEGIN_CAST]], ptr %[[BEGIN_ADDR]]
// LLVM:   %[[RANGE:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// LLVM:   %[[RANGE_CAST:.*]] = getelementptr i32, ptr %[[RANGE]], i32 0
// LLVM:   %[[END_PTR:.*]] = getelementptr i32, ptr %[[RANGE_CAST]], i64 4
// LLVM:   store ptr %[[END_PTR]], ptr %[[END_ADDR]]
// LLVM:   br label %[[COND:.*]]
// LLVM: [[COND]]:
// LLVM:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// LLVM:   %[[END:.*]] = load ptr, ptr %[[END_ADDR]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[BEGIN]], %[[END]]
// LLVM:   br i1 %[[CMP]], label %[[BODY:.*]], label %[[END:.*]]
// LLVM: [[BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// LLVM:   %[[ARR_CUR:.*]] = load i32, ptr %[[CUR]]
// LLVM:   store i32 %[[ARR_CUR]], ptr %[[X_ADDR]]
// LLVM:   br label %[[STEP:.*]]
// LLVM: [[STEP]]:
// LLVM:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// LLVM:   %[[NEXT:.*]] = getelementptr i32, ptr %[[BEGIN]], i64 1
// LLVM:   store ptr %[[NEXT]], ptr %[[BEGIN_ADDR]]
// LLVM:   br label %[[COND]]
// LLVM: [[END]]:
// LLVM:   br label %[[EXIT:.*]]
// LLVM: [[EXIT]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z2l5v()
// OGCG:   %[[ARR_ADDR:.*]] = alloca [4 x i32]
// OGCG:   %[[RANGE_ADDR:.*]] = alloca ptr
// OGCG:   %[[BEGIN_ADDR:.*]] = alloca ptr
// OGCG:   %[[END_ADDR:.*]] = alloca ptr
// OGCG:   %[[X_ADDR:.*]] = alloca i32
// OGCG:   call void @llvm.memcpy.p0.p0.i64
// OGCG:   store ptr %[[ARR_ADDR]], ptr %[[RANGE_ADDR]]
// OGCG:   %[[BEGIN:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// OGCG:   %[[BEGIN_CAST:.*]] = getelementptr inbounds [4 x i32], ptr %[[BEGIN]], i64 0, i64 0
// OGCG:   store ptr %[[BEGIN_CAST]], ptr %[[BEGIN_ADDR]]
// OGCG:   %[[RANGE:.*]] = load ptr, ptr %[[RANGE_ADDR]]
// OGCG:   %[[RANGE_CAST:.*]] = getelementptr inbounds [4 x i32], ptr %[[RANGE]], i64 0, i64 0
// OGCG:   %[[END_PTR:.*]] = getelementptr inbounds i32, ptr %[[RANGE_CAST]], i64 4
// OGCG:   store ptr %[[END_PTR]], ptr %[[END_ADDR]]
// OGCG:   br label %[[COND:.*]]
// OGCG: [[COND]]:
// OGCG:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// OGCG:   %[[END:.*]] = load ptr, ptr %[[END_ADDR]]
// OGCG:   %[[CMP:.*]] = icmp ne ptr %[[BEGIN]], %[[END]]
// OGCG:   br i1 %[[CMP]], label %[[BODY:.*]], label %[[END:.*]]
// OGCG: [[BODY]]:
// OGCG:   %[[CUR:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// OGCG:   %[[ARR_CUR:.*]] = load i32, ptr %[[CUR]]
// OGCG:   store i32 %[[ARR_CUR]], ptr %[[X_ADDR]]
// OGCG:   br label %[[STEP:.*]]
// OGCG: [[STEP]]:
// OGCG:   %[[BEGIN:.*]] = load ptr, ptr %[[BEGIN_ADDR]]
// OGCG:   %[[NEXT:.*]] = getelementptr inbounds nuw i32, ptr %[[BEGIN]], i32 1
// OGCG:   store ptr %[[NEXT]], ptr %[[BEGIN_ADDR]]
// OGCG:   br label %[[COND]]
// OGCG: [[END]]:
// OGCG:   ret void

void test_do_while_false() {
  do {
  } while (0);
}

// CIR: cir.func{{.*}} @_Z19test_do_while_falsev()
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     cir.do {
// CIR-NEXT:       cir.yield
// CIR-NEXT:     } while {
// CIR-NEXT:       %[[ZERO:.*]] = cir.const #cir.int<0> : !s32i
// CIR-NEXT:       %[[FALSE:.*]] = cir.cast int_to_bool %[[ZERO]] : !s32i -> !cir.bool
// CIR-NEXT:       cir.condition(%[[FALSE]])

// LLVM: define{{.*}} void @_Z19test_do_while_falsev()
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL3:.*]]
// LLVM: [[LABEL2:.*]]:
// LLVM:   br i1 false, label %[[LABEL3]], label %[[LABEL4:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL5:.*]]
// LLVM: [[LABEL5]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z19test_do_while_falsev()
// OGCG: entry:
// OGCG:   br label %[[DO_BODY:.*]]
// OGCG: [[DO_BODY]]:
// OGCG:   br label %[[DO_END:.*]]
// OGCG: [[DO_END]]:
// OGCG:   ret void

void test_empty_while_true() {
  while (true) {
    return;
  }
}

// CIR: cir.func{{.*}} @_Z21test_empty_while_truev()
// CIR-NEXT:   cir.scope {
// CIR-NEXT:     cir.while {
// CIR-NEXT:       %[[TRUE:.*]] = cir.const #true
// CIR-NEXT:       cir.condition(%[[TRUE]])
// CIR-NEXT:     } do {
// CIR-NEXT:       cir.scope {
// CIR-NEXT:         cir.return
// CIR-NEXT:       }
// CIR-NEXT:       cir.yield

// LLVM: define{{.*}} void @_Z21test_empty_while_truev()
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL6:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4]]
// LLVM: [[LABEL4]]:
// LLVM:   ret void
// LLVM: [[LABEL5:.*]]:
// LLVM-SAME: ; No predecessors!
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL7]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z21test_empty_while_truev()
// OGCG: entry:
// OGCG:   br label %[[WHILE_BODY:.*]]
// OGCG: [[WHILE_BODY]]:
// OGCG:   ret void

void unreachable_after_continue() {
  for (;;) {
    continue;
    int x = 1;
  }
}

// CIR: cir.func{{.*}} @_Z26unreachable_after_continuev()
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.condition(%[[TRUE]])
// CIR:     } body {
// CIR:       cir.scope {
// CIR:         %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR:         cir.continue
// CIR:       ^bb1:  // no predecessors
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:         cir.store{{.*}} %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } step {
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define{{.*}} void @_Z26unreachable_after_continuev()
// LLVM:   %[[X:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL8:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL5:.*]]:
// LLVM-SAME: ; No predecessors!
// LLVM:   store i32 1, ptr %[[X]], align 4
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL7]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL8]]:
// LLVM:   br label %[[LABEL9:]]
// LLVM: [[LABEL9]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z26unreachable_after_continuev()
// OGCG: entry:
// OGCG:   %[[X:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_COND]]

void unreachable_after_break() {
  for (;;) {
    break;
    int x = 1;
  }
}

// CIR: cir.func{{.*}} @_Z23unreachable_after_breakv()
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:       %[[TRUE:.*]] = cir.const #true
// CIR:       cir.condition(%[[TRUE]])
// CIR:     } body {
// CIR:       cir.scope {
// CIR:         %[[X:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init] {alignment = 4 : i64}
// CIR:         cir.break
// CIR:       ^bb1:  // no predecessors
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR:         cir.store{{.*}} %[[ONE]], %[[X]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.yield
// CIR:       }
// CIR:       cir.yield
// CIR:     } step {
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.return
// CIR: }

// LLVM: define{{.*}} void @_Z23unreachable_after_breakv()
// LLVM:   %[[X:.*]] = alloca i32, i64 1, align 4
// LLVM:   br label %[[LABEL1:.*]]
// LLVM: [[LABEL1]]:
// LLVM:   br label %[[LABEL2:.*]]
// LLVM: [[LABEL2]]:
// LLVM:   br i1 true, label %[[LABEL3:.*]], label %[[LABEL8:.*]]
// LLVM: [[LABEL3]]:
// LLVM:   br label %[[LABEL4:.*]]
// LLVM: [[LABEL4]]:
// LLVM:   br label %[[LABEL8]]
// LLVM: [[LABEL5:.*]]:
// LLVM-SAME: ; No predecessors!
// LLVM:   store i32 1, ptr %[[X]], align 4
// LLVM:   br label %[[LABEL6:.*]]
// LLVM: [[LABEL6]]:
// LLVM:   br label %[[LABEL7:.*]]
// LLVM: [[LABEL7]]:
// LLVM:   br label %[[LABEL2]]
// LLVM: [[LABEL8]]:
// LLVM:   br label %[[LABEL9:]]
// LLVM: [[LABEL9]]:
// LLVM:   ret void

// OGCG: define{{.*}} void @_Z23unreachable_after_breakv()
// OGCG: entry:
// OGCG:   %[[X:.*]] = alloca i32, align 4
// OGCG:   br label %[[FOR_COND:.*]]
// OGCG: [[FOR_COND]]:
// OGCG:   br label %[[FOR_END:.*]]
// OGCG: [[FOR_END]]:
// OGCG:   ret void
