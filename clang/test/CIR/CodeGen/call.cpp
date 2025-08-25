// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

void f1() {}
void f2() {
  f1();
}

// CIR-LABEL: cir.func{{.*}} @_Z2f1v
// CIR-LABEL: cir.func{{.*}} @_Z2f2v
// CIR:         cir.call @_Z2f1v() : () -> ()

// LLVM-LABEL: define{{.*}} void @_Z2f2v() {
// LLVM:         call void @_Z2f1v()

int f3() { return 2; }
int f4() {
  int x = f3();
  return x;
}

// CIR-LABEL: cir.func{{.*}} @_Z2f3v() -> !s32i
// CIR-LABEL: cir.func{{.*}} @_Z2f4v() -> !s32i
// CIR:         cir.call @_Z2f3v() : () -> !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z2f4v() {
// LLVM:         %{{.+}} = call i32 @_Z2f3v()

int f5(int a, int *b, bool c);
int f6() {
  int b = 1;
  return f5(2, &b, false);
}

// CIR-LABEL: cir.func{{.*}} @_Z2f6v() -> !s32i
// CIR:         %[[#b:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:         %[[#a:]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:    %[[#c:]] = cir.const #false
// CIR-NEXT:    %{{.+}} = cir.call @_Z2f5iPib(%[[#a]], %[[#b:]], %[[#c]]) : (!s32i, !cir.ptr<!s32i>, !cir.bool) -> !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z2f6v() {
// LLVM:         %{{.+}} = call i32 @_Z2f5iPib(i32 2, ptr %{{.+}}, i1 false)

int f7(int (*ptr)(int, int)) {
  return ptr(1, 2);
}

// CIR-LABEL: cir.func{{.*}} @_Z2f7PFiiiE
// CIR:         %[[#ptr:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>>, !cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>
// CIR-NEXT:    %[[#a:]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:    %[[#b:]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:    %{{.+}} = cir.call %[[#ptr]](%[[#a]], %[[#b]]) : (!cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>, !s32i, !s32i) -> !s32i

// LLVM-LABEL: define{{.*}} i32 @_Z2f7PFiiiE
// LLVM:         %[[#ptr:]] = load ptr, ptr %{{.+}}
// LLVM-NEXT:    %{{.+}} = call i32 %[[#ptr]](i32 1, i32 2)

void f8(int a, ...);
void f9() {
  f8(1);
  f8(1, 2, 3, 4);
}

// CIR-LABEL: cir.func{{.*}} @_Z2f9v()
// CIR:         cir.call @_Z2f8iz(%{{.+}}) : (!s32i) -> ()
// CIR:         cir.call @_Z2f8iz(%{{.+}}, %{{.+}}, %{{.+}}, %{{.+}}) : (!s32i, !s32i, !s32i, !s32i) -> ()

// LLVM-LABEL: define{{.*}} void @_Z2f9v()
// LLVM:         call void (i32, ...) @_Z2f8iz(i32 1)
// LLVM:         call void (i32, ...) @_Z2f8iz(i32 1, i32 2, i32 3, i32 4)

struct S {
  int x;
  int y;
};

S f10();
void f11() {
  S s = f10();
}

// CIR-LABEL: cir.func{{.*}} @_Z3f11v()
// CIR:         %[[#s:]] = cir.call @_Z3f10v() : () -> !rec_S
// CIR-NEXT:    cir.store align(4) %[[#s]], %{{.+}} : !rec_S, !cir.ptr<!rec_S>

// LLVM-LABEL: define{{.*}} void @_Z3f11v()
// LLVM:         %[[#s:]] = call %struct.S @_Z3f10v()
// LLVM-NEXT:    store %struct.S %[[#s]], ptr %{{.+}}, align 4

void f12() {
  f10();
}

// CIR-LABEL: cir.func{{.*}} @_Z3f12v()
// CIR:         %[[#slot:]] = cir.alloca !rec_S, !cir.ptr<!rec_S>, ["agg.tmp0"]
// CIR-NEXT:    %[[#ret:]] = cir.call @_Z3f10v() : () -> !rec_S
// CIR-NEXT:    cir.store align(4) %[[#ret]], %[[#slot]] : !rec_S, !cir.ptr<!rec_S>

// LLVM-LABEL: define{{.*}} void @_Z3f12v() {
// LLVM:         %[[#slot:]] = alloca %struct.S, i64 1, align 4
// LLVM-NEXT:    %[[#ret:]] = call %struct.S @_Z3f10v()
// LLVM-NEXT:    store %struct.S %[[#ret]], ptr %[[#slot]], align 4

void f13() noexcept;
void f14() {
  f13();
}

// CIR-LABEL: cir.func{{.+}} @_Z3f14v()
// CIR:         cir.call @_Z3f13v() nothrow : () -> ()
// CIR:       }

// LLVM-LABEL: define{{.+}} void @_Z3f14v()
// LLVM:         call void @_Z3f13v() #[[LLVM_ATTR_0:.+]]
// LLVM:       }

int f15();
void f16() {
  using T = int;
  f15().~T();
}

// CIR-LABEL: @_Z3f16v
// CIR-NEXT:    %{{.+}} = cir.call @_Z3f15v() : () -> !s32i
// CIR:       }

// LLVM-LABEL: define{{.+}} void @_Z3f16v() {
// LLVM-NEXT:    %{{.+}} = call i32 @_Z3f15v()
// LLVM:       }

// LLVM: attributes #[[LLVM_ATTR_0]] = { nounwind }
