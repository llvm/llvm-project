// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

void foo();

void discarded() {
  void{};
}

// CIR-LABEL: cir.func{{.*}}@_Z9discardedv()
// CIR-NEXT: cir.return
// LLVM-LABEL: define{{.*}} void @_Z9discardedv()
// LLVM-NOT: {{store|call}}
// LLVM: ret void

void returned() { return void{}; }

// CIR-LABEL: cir.func{{.*}}@_Z8returnedv()
// CIR-NEXT: cir.return
// LLVM-LABEL: define{{.*}} void @_Z8returnedv()
// LLVM-NOT: {{store|call}}
// LLVM: ret void

void comma() { void{}, foo(); }

// CIR-LABEL: cir.func{{.*}}@_Z5commav()
// CIR-NEXT: cir.call @_Z3foov() : () -> ()
// CIR-NEXT: cir.return
// LLVM-LABEL: define{{.*}} void @_Z5commav()
// LLVM: call void @_Z3foov()
// LLVM-NEXT: ret void

template <class T> T valueInit() { return T{}; }
void instantiate() { valueInit<void>(); }

// CIR-LABEL: cir.func{{.*}}@_Z11instantiatev()
// CIR-NEXT: cir.call @_Z9valueInitIvET_v() : () -> ()
// CIR-NEXT: cir.return
// CIR-LABEL: cir.func{{.*}}@_Z9valueInitIvET_v()
// CIR-NEXT: cir.return
// LLVM-LABEL: define{{.*}} void @_Z11instantiatev()
// LLVM: call void @_Z9valueInitIvET_v()
// LLVM-LABEL: define{{.*}} void @_Z9valueInitIvET_v()
// LLVM-NOT: {{store|call}}
// LLVM: ret void

void ternaryCheap(bool b) {
  b ? void() : void();
  b ? (void)0 : (void)1;
  b ? void{} : void{};
}

// CIR-LABEL: cir.func{{.*}}@_Z12ternaryCheapb
// CIR: %[[ARG:.*]] = cir.alloca {{.*}} : !cir.ptr<!cir.bool>
// CIR: cir.load{{.*}}%[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.load{{.*}}%[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.load{{.*}}%[[ARG]] : !cir.ptr<!cir.bool>, !cir.bool
// CIR-NEXT: cir.return
// LLVM-LABEL: define{{.*}} void @_Z12ternaryCheapb
// LLVM-NOT: {{select|br}}
// LLVM: load i8, ptr %{{.+}}, align 1
// LLVM-NOT: {{select|br}}
// LLVM: load i8, ptr %{{.+}}, align 1
// LLVM-NOT: {{select|br}}
// LLVM: load i8, ptr %{{.+}}, align 1
// LLVM-NOT: {{select|br}}
// LLVM: ret void

void ternaryMixed(bool b) {
  b ? void{} : foo();
}

// CIR-LABEL: cir.func{{.*}}@_Z12ternaryMixedb
// CIR: cir.ternary(%{{.*}}, true {
// CIR-NEXT: cir.yield
// CIR-NEXT: }, false {
// CIR-NEXT: cir.call @_Z3foov()
// CIR-NEXT: cir.yield
// CIR-NEXT: }) : (!cir.bool) -> ()
// LLVM-LABEL: define{{.*}} void @_Z12ternaryMixedb
// LLVM: br i1 %{{.*}}, label %[[TRUE:.*]], label %[[FALSE:.*]]
// LLVM: [[TRUE]]:
// LLVM-NEXT: br
// LLVM: [[FALSE]]:
// LLVM-NEXT: call void @_Z3foov()
// LLVM-NEXT: br
// LLVM: ret void

void folded() { true ? void{} : foo(); }

// CIR-LABEL: cir.func{{.*}}@_Z6foldedv()
// CIR-NEXT: cir.return
// LLVM-LABEL: define{{.*}} void @_Z6foldedv()
// LLVM-NOT: call
// LLVM: ret void
