// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

void f1() {}
void f2() {
  f1();
}

// CIR-LABEL: cir.func @_Z2f1v
// CIR-LABEL: cir.func @_Z2f2v
// CIR:         cir.call @_Z2f1v() : () -> ()

// LLVM-LABEL: define void @_Z2f2v() {
// LLVM:         call void @_Z2f1v()

int f3() { return 2; }
int f4() {
  int x = f3();
  return x;
}

// CIR-LABEL: cir.func @_Z2f3v() -> !s32i
// CIR-LABEL: cir.func @_Z2f4v() -> !s32i
// CIR:         cir.call @_Z2f3v() : () -> !s32i

// LLVM-LABEL: define i32 @_Z2f4v() {
// LLVM:         %{{.+}} = call i32 @_Z2f3v()

int f5(int a, int *b, bool c);
int f6() {
  int b = 1;
  return f5(2, &b, false);
}

// CIR-LABEL: cir.func @_Z2f6v() -> !s32i
// CIR:         %[[#b:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["b", init]
// CIR:         %[[#a:]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:    %[[#c:]] = cir.const #false
// CIR-NEXT:    %{{.+}} = cir.call @_Z2f5iPib(%[[#a]], %[[#b:]], %[[#c]]) : (!s32i, !cir.ptr<!s32i>, !cir.bool) -> !s32i

// LLVM-LABEL: define i32 @_Z2f6v() {
// LLVM:         %{{.+}} = call i32 @_Z2f5iPib(i32 2, ptr %{{.+}}, i1 false)

int f7(int (*ptr)(int, int)) {
  return ptr(1, 2);
}

// CIR-LABEL: cir.func @_Z2f7PFiiiE
// CIR:         %[[#ptr:]] = cir.load{{.*}} %{{.+}} : !cir.ptr<!cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>>, !cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>
// CIR-NEXT:    %[[#a:]] = cir.const #cir.int<1> : !s32i
// CIR-NEXT:    %[[#b:]] = cir.const #cir.int<2> : !s32i
// CIR-NEXT:    %{{.+}} = cir.call %[[#ptr]](%[[#a]], %[[#b]]) : (!cir.ptr<!cir.func<(!s32i, !s32i) -> !s32i>>, !s32i, !s32i) -> !s32i

// LLVM-LABEL: define i32 @_Z2f7PFiiiE
// LLVM:         %[[#ptr:]] = load ptr, ptr %{{.+}}
// LLVM-NEXT:    %{{.+}} = call i32 %[[#ptr]](i32 1, i32 2)
