// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

class A {
public:
  static char *b(int);
};

int h=0;

class F {
public:
  const char *b();
  A g;
};

const char *F::b() { return g.b(h); }

void fn1() { F f1; }

// CIR: cir.func {{.*}} @_ZN1F1bEv
// CIR:   %[[H_PTR:.*]] = cir.get_global @h : !cir.ptr<!s32i>
// CIR:   %[[H_VAL:.*]] = cir.load{{.*}} %[[H_PTR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[RET:.*]] = cir.call @_ZN1A1bEi(%[[H_VAL]]) : (!s32i) -> !cir.ptr<!s8i>

// LLVM: define {{.*}} ptr @_ZN1F1bEv
// LLVM:   %[[VAR_H:.*]] = load i32, ptr @h
// LLVM:   %[[RET:.*]] = call ptr @_ZN1A1bEi(i32 %[[VAR_H]])

// OGCG: define {{.*}} ptr @_ZN1F1bEv
// OGCG:   %[[VAR_H:.*]] = load i32, ptr @h
// OGCG:   %[[RET:.*]] = call noundef ptr @_ZN1A1bEi(i32 noundef %[[VAR_H]])

class B {
public:
  B();
  int (&indirect_callee_int_ref)(int);
};

class C {
public:
  int call_indirect(int v) { return inner.indirect_callee_int_ref(v); };
  B inner;
};

void fn2() { C c1; c1.call_indirect(2); }

// CIR: cir.func {{.*}} @_ZN1C13call_indirectEi(%[[THIS_ARG:.*]]: !cir.ptr<!rec_C> {{.*}}, %[[V_ARG:.*]]: !s32i {{.*}}) -> !s32i
// CIR:   %[[THIS_ADDR:.*]] = cir.alloca !cir.ptr<!rec_C>, !cir.ptr<!cir.ptr<!rec_C>>, ["this", init]
// CIR:   %[[V_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["v", init]
// CIR:   cir.store %[[THIS_ARG]], %[[THIS_ADDR]]
// CIR:   cir.store %[[V_ARG]], %[[V_ADDR]]
// CIR:   %[[THIS:.*]] = cir.load %[[THIS_ADDR]]
// CIR:   %[[INNER:.*]] = cir.get_member %[[THIS]][0] {name = "inner"} : !cir.ptr<!rec_C> -> !cir.ptr<!rec_B>
// CIR:   %[[INDIRECT_CALLEE_PTR:.*]] = cir.get_member %[[INNER]][0] {name = "indirect_callee_int_ref"}
// CIR:   %[[INDIRECT_CALLEE:.*]] = cir.load %[[INDIRECT_CALLEE_PTR]]
// CIR:   %[[V:.*]] = cir.load{{.*}} %[[V_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR:   %[[RET:.*]] = cir.call %[[INDIRECT_CALLEE]](%[[V]])

// LLVM: define {{.*}} i32 @_ZN1C13call_indirectEi(ptr %[[THIS_ARG:.*]], i32 %[[V_ARG:.*]])
// LLVM:   %[[THIS_ADDR:.*]] = alloca ptr
// LLVM:   %[[V_ADDR:.*]] = alloca i32
// LLVM:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// LLVM:   store i32 %[[V_ARG]], ptr %[[V_ADDR]]
// LLVM:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// LLVM:   %[[INNER:.*]] = getelementptr %class.C, ptr %[[THIS]], i32 0, i32 0
// LLVM:   %[[INDIRECT_CALLEE_PTR:.*]] = getelementptr %class.B, ptr %[[INNER]], i32 0, i32 0
// LLVM:   %[[INDIRECT_CALLEE:.*]] = load ptr, ptr %[[INDIRECT_CALLEE_PTR]]
// LLVM:   %[[V:.*]] = load i32, ptr %[[V_ADDR]]
// LLVM:   %[[RET:.*]] = call i32 %[[INDIRECT_CALLEE]](i32 %[[V]])

// OGCG: define {{.*}} i32 @_ZN1C13call_indirectEi(ptr {{.*}} %[[THIS_ARG:.*]], i32 {{.*}} %[[V_ARG:.*]])
// OGCG:   %[[THIS_ADDR:.*]] = alloca ptr
// OGCG:   %[[V_ADDR:.*]] = alloca i32
// OGCG:   store ptr %[[THIS_ARG]], ptr %[[THIS_ADDR]]
// OGCG:   store i32 %[[V_ARG]], ptr %[[V_ADDR]]
// OGCG:   %[[THIS:.*]] = load ptr, ptr %[[THIS_ADDR]]
// OGCG:   %[[INNER:.*]] = getelementptr inbounds nuw %class.C, ptr %[[THIS]], i32 0, i32 0
// OGCG:   %[[INDIRECT_CALLEE_PTR:.*]] = getelementptr inbounds nuw %class.B, ptr %[[INNER]], i32 0, i32 0
// OGCG:   %[[INDIRECT_CALLEE:.*]] = load ptr, ptr %[[INDIRECT_CALLEE_PTR]]
// OGCG:   %[[V:.*]] = load i32, ptr %[[V_ADDR]]
// OGCG:   %[[RET:.*]] = call noundef i32 %[[INDIRECT_CALLEE]](i32 noundef %[[V]])
