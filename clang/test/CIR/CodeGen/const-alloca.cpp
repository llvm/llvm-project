// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

int produce_int();
void blackbox(const int &);

void local_const_int() {
  const int x = produce_int();
}

// CIR-LABEL: @_Z15local_const_intv
// CIR:   %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR: }

void param_const_int(const int x) {}

// CIR-LABEL: @_Z15param_const_inti
// CIR:  %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR: }

void local_constexpr_int() {
  constexpr int x = 42;
  blackbox(x);
}

// CIR-LABEL: @_Z19local_constexpr_intv
// CIR:   %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR: }

void local_reference() {
  int x = 0;
  int &r = x;
}

// CIR-LABEL: @_Z15local_referencev
// CIR:   %{{.+}} = cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const]
// CIR: }

struct Foo {
  int a;
  int b;
};

Foo produce_foo();

void local_const_struct() {
  const Foo x = produce_foo();
}

// CIR-LABEL: @_Z18local_const_structv
// CIR:   %{{.+}} = cir.alloca !ty_Foo, !cir.ptr<!ty_Foo>, ["x", init, const]
// CIR: }

[[clang::optnone]]
int local_const_load_store() {
  const int x = produce_int();
  int y = x;
  return y;
}

// CIR-LABEL: @_Z22local_const_load_storev
// CIR: %{{.+}} = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const] {alignment = 4 : i64}
// CIR: }

// LLVM-LABEL: @_Z22local_const_load_storev
//      LLVM: %[[#INIT:]] = call i32 @_Z11produce_intv()
// LLVM-NEXT: store i32 %[[#INIT]], ptr %[[#SLOT:]], align 4, !tbaa !{{.*}}, !invariant.group !{{.+}}
// LLVM-NEXT: %{{.+}} = load i32, ptr %[[#SLOT]], align 4, !tbaa !{{.*}}, !invariant.group !{{.+}}
// LLVM: }

int local_const_optimize() {
  const int x = produce_int();
  blackbox(x);
  blackbox(x);
  return x;
}

// LLVM-LABEL: @_Z20local_const_optimizev()
// LLVM-NEXT:    %[[#slot:]] = alloca i32, align 4
// LLVM-NEXT:    %[[#init:]] = tail call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[#init]], ptr %[[#slot]], align 4, !tbaa !{{.*}}, !invariant.group !{{.+}}
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr nonnull %[[#slot]])
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr nonnull %[[#slot]])
// LLVM-NEXT:    ret i32 %[[#init]]
// LLVM-NEXT:  }
