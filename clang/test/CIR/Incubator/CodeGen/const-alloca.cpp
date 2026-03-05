// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir --check-prefix=CIR %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll --check-prefix=LLVM %s

int produce_int();
void blackbox(const int &);
void consume(int);

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
// CIR:   %{{.+}} = cir.alloca !rec_Foo, !cir.ptr<!rec_Foo>, ["x", init, const]
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

int local_scoped_const() {
  {
    const int x = produce_int();
    blackbox(x);
    return x;
  }
}

// CIR-LABEL: @_Z18local_scoped_constv()
//      CIR:    cir.scope {
// CIR-NEXT:      %[[#x_slot:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT:      %[[#init:]] = cir.call @_Z11produce_intv() : () -> !s32i
// CIR-NEXT:      cir.store{{.*}} %[[#init]], %[[#x_slot]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      cir.call @_Z8blackboxRKi(%[[#x_slot]]) : (!cir.ptr<!s32i>) -> ()
// CIR-NEXT:      %[[#x_reload:]] = cir.load{{.*}} %[[#x_slot]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:      cir.store{{.*}} %[[#x_reload]], %[[#ret_slot:]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      %[[#ret:]] = cir.load{{.*}} %[[#ret_slot]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:      cir.return %[[#ret]] : !s32i
// CIR-NEXT:    }
//      CIR:  }

// LLVM-LABEL: @_Z18local_scoped_constv()
// LLVM-NEXT:    %[[#x_slot:]] = alloca i32, align 4
// LLVM-NEXT:    %[[#init:]] = tail call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[#init]], ptr %[[#x_slot]], align 4, !tbaa !{{.+}}, !invariant.group !{{.+}}
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr nonnull %[[#x_slot]])
// LLVM-NEXT:    ret i32 %[[#init]]
// LLVM-NEXT:  }

void local_const_in_loop() {
  for (int i = 0; i < 10; ++i) {
    const int x = produce_int();
    blackbox(x);
    consume(x);
  }
}

// CIR-LABEL: @_Z19local_const_in_loopv
//      CIR:    cir.scope {
//      CIR:      cir.for : cond {
//      CIR:      } body {
// CIR-NEXT:        cir.scope {
// CIR-NEXT:          %[[#x_slot:]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT:          %[[#init:]] = cir.call @_Z11produce_intv() : () -> !s32i
// CIR-NEXT:          cir.store{{.*}} %[[#init]], %[[#x_slot]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:          cir.call @_Z8blackboxRKi(%[[#x_slot]]) : (!cir.ptr<!s32i>) -> ()
// CIR-NEXT:          %[[#x_reload:]] = cir.load{{.*}} %[[#x_slot]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:          cir.call @_Z7consumei(%[[#x_reload]]) : (!s32i) -> ()
// CIR-NEXT:        }
// CIR-NEXT:        cir.yield
// CIR-NEXT:      } step {
//      CIR:      }
// CIR-NEXT:    }
// CIR-NEXT:    cir.return
// CIR-NEXT:  }

// LLVM-LABEL: @_Z19local_const_in_loopv()
//      LLVM:    %[[#x_ptr:]] = call ptr @llvm.launder.invariant.group.p0(ptr nonnull %1)
// LLVM-NEXT:    %[[#init:]] = call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[#init]], ptr %[[#x_ptr]], align 4, !tbaa !{{.+}}, !invariant.group !{{.+}}
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr nonnull %[[#x_ptr]])
// LLVM-NEXT:    call void @_Z7consumei(i32 %[[#init]])
//      LLVM:  }

void local_const_in_while_condition() {
  while (const int x = produce_int()) {
    blackbox(x);
  }
}

// LLVM-LABEL: @_Z30local_const_in_while_conditionv()
//      LLVM:    %[[#x_slot:]] = alloca i32, align 4
// LLVM-NEXT:    %[[#init:]] = tail call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[#init]], ptr %[[#x_slot]], align 4
// LLVM-NEXT:    %[[loop_cond:.+]] = icmp eq i32 %[[#init]], 0
// LLVM-NEXT:    br i1 %[[loop_cond]], label %{{.+}}, label %[[loop_body:.+]]
//      LLVM:  [[loop_body]]:
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr nonnull %[[#x_slot]])
// LLVM-NEXT:    %[[#next:]] = call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[#next]], ptr %[[#x_slot]], align 4
// LLVM-NEXT:    %[[cond:.+]] = icmp eq i32 %[[#next]], 0
// LLVM-NEXT:    br i1 %[[cond]], label %{{.+}}, label %[[loop_body]]
//      LLVM:  }
