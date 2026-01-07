// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
//
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir -O1
// RUN: cir-opt -cir-hoist-allocas -o %t-opt.cir %t.cir
// RUN: FileCheck --input-file=%t-opt.cir %s --check-prefix=CIR-OPT
//
// RUN: cir-translate -o %t-llvm.ll %t-opt.cir -cir-to-llvmir --disable-cc-lowering
// RUN: FileCheck --input-file=%t-llvm.ll %s --check-prefix=LLVM
//
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -O1 -emit-llvm %s -o %t-og.ll
// RUN: FileCheck --input-file=%t-og.ll %s --check-prefix=OGCG

int produce_int();
void blackbox(const int &);
void consume(int);

void local_const_int() {
  const int x = produce_int();
}

// CIR-LABEL: @_Z15local_const_intv
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT: cir.call @_Z11produce_intv()
// CIR-NEXT: cir.store
// CIR-NEXT: cir.return
// CIR-NEXT: }

// CIR-OPT-LABEL: @_Z15local_const_intv
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-OPT-NEXT: cir.call @_Z11produce_intv()
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.return
// CIR-OPT-NEXT: }

// LLVM-LABEL: @_Z15local_const_intv
// LLVM-NEXT: %[[X:.*]] = alloca i32
// LLVM-NEXT: %[[RES:.*]] = call i32 @_Z11produce_intv()
// LLVM-NEXT: store i32 %[[RES]], ptr %[[X]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT: ret void
// LLVM-NEXT: }

// OGCG-LABEL: @_Z15local_const_intv
// OGCG-NEXT: entry:
// OGCG-NEXT: tail call {{.*}}i32 @_Z11produce_intv()
// OGCG-NEXT: ret void
// OGCG-NEXT: }

void param_const_int(const int x) {}

// CIR-LABEL: @_Z15param_const_inti
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT: cir.store
// CIR-NEXT: cir.return
// CIR-NEXT: }

// CIR-OPT-LABEL: @_Z15param_const_inti
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.return
// CIR-OPT-NEXT: }

// LLVM-LABEL: @_Z15param_const_inti
// LLVM-SAME: (i32 %[[PARM:.*]])
// LLVM-NEXT: %[[X:.*]] = alloca i32
// LLVM-NEXT: store i32 %[[PARM]], ptr %[[X]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT: ret void
// LLVM-NEXT: }

// OGCG-LABEL: @_Z15param_const_inti
// OGCG-NEXT: entry:
// OGCG-NEXT: ret void
// OGCG-NEXT: }

void local_constexpr_int() {
  constexpr int x = 42;
  blackbox(x);
}

// CIR-LABEL: @_Z19local_constexpr_intv
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT: cir.const
// CIR-NEXT: cir.store
// CIR-NEXT: cir.call @_Z8blackboxRKi(%{{.+}})
// CIR-NEXT: cir.return
// CIR-NEXT: }

// CIR-OPT-LABEL: @_Z19local_constexpr_intv
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-OPT-NEXT: cir.const
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.call @_Z8blackboxRKi(%{{.+}})
// CIR-OPT-NEXT: cir.return
// CIR-OPT-NEXT: }

// LLVM-LABEL: @_Z19local_constexpr_intv
// LLVM-NEXT: %[[X:.*]] = alloca i32
// LLVM-NEXT: store i32 42, ptr %[[X]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT: call void @_Z8blackboxRKi(ptr %[[X]])
// LLVM-NEXT: ret void
// LLVM-NEXT: }

// OGCG-LABEL: @_Z19local_constexpr_intv
// OGCG-NEXT: entry:
// OGCG-NEXT: %[[X:.*]] = alloca i32
// OGCG-NEXT: call void @llvm.lifetime.start
// OGCG-NEXT: store i32 42, ptr %[[X]], align 4
// OGCG-NEXT: call void @_Z8blackboxRKi(ptr {{.*}}%[[X]])
// OGCG-NEXT: call void @llvm.lifetime.end
// OGCG-NEXT: ret void
// OGCG-NEXT: }

void local_reference() {
  int x = 0;
  int &r = x;
}

// CIR-LABEL: @_Z15local_referencev
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR-NEXT: cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const]
// CIR-NEXT: cir.const
// CIR-NEXT: cir.store
// CIR-NEXT: cir.store
// CIR-NEXT: cir.return
// CIR-NEXT: }

// CIR-OPT-LABEL: @_Z15local_referencev
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR-OPT-NEXT: cir.alloca !cir.ptr<!s32i>, !cir.ptr<!cir.ptr<!s32i>>, ["r", init, const]
// CIR-OPT-NEXT: cir.const
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.return
// CIR-OPT-NEXT: }

// LLVM-LABEL: @_Z15local_referencev
// LLVM-NEXT: %[[X:.*]] = alloca i32
// LLVM-NEXT: %[[R:.*]] = alloca ptr
// LLVM-NEXT: store i32 0, ptr %[[X]]
// LLVM-NEXT: store ptr %[[X]], ptr %[[R]], align 8, !invariant.group !{{.+}}
// LLVM-NEXT: ret void
// LLVM-NEXT: }

// OGCG-LABEL: @_Z15local_referencev
// OGCG-NEXT: entry:
// OGCG-NEXT: ret void
// OGCG-NEXT: }

struct Foo {
  int a;
  int b;
};

Foo produce_foo();

void local_const_struct() {
  const Foo x = produce_foo();
}

// CIR-LABEL: @_Z18local_const_structv
// CIR-NEXT: cir.alloca !rec_Foo, !cir.ptr<!rec_Foo>, ["x", init, const]
// CIR-NEXT: cir.call @_Z11produce_foov()
// CIR-NEXT: cir.store
// CIR-NEXT: cir.return
// CIR-NEXT: }

// CIR-OPT-LABEL: @_Z18local_const_structv
// CIR-OPT-NEXT: cir.alloca !rec_Foo, !cir.ptr<!rec_Foo>, ["x", init, const]
// CIR-OPT-NEXT: cir.call @_Z11produce_foov()
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.return
// CIR-OPT-NEXT: }

// LLVM-LABEL: @_Z18local_const_structv
// LLVM-NEXT: %[[X:.*]] = alloca %struct.Foo
// LLVM-NEXT: %[[RES:.*]] = call %struct.Foo @_Z11produce_foov()
// LLVM-NEXT: store %struct.Foo %[[RES]], ptr %[[X]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT: ret void
// LLVM-NEXT: }

// OGCG-LABEL: @_Z18local_const_structv
// OGCG-NEXT: entry:
// OGCG-NEXT: tail call i64 @_Z11produce_foov()
// OGCG-NEXT: ret void
// OGCG-NEXT: }

[[clang::optnone]]
int local_const_load_store() {
  const int x = produce_int();
  int y = x;
  return y;
}

// CIR-LABEL: @_Z22local_const_load_storev
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR-NEXT: cir.call @_Z11produce_intv()
// CIR-NEXT: cir.store
// CIR-NEXT: cir.load
// CIR-NEXT: cir.store
// CIR-NEXT: cir.load
// CIR-NEXT: cir.store
// CIR-NEXT: cir.load
// CIR-NEXT: cir.return
// CIR-NEXT: }

// CIR-OPT-LABEL: @_Z22local_const_load_storev
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["y", init]
// CIR-OPT-NEXT: cir.call @_Z11produce_intv()
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.load
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.load
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.load
// CIR-OPT-NEXT: cir.return
// CIR-OPT-NEXT: }

// LLVM-LABEL: @_Z22local_const_load_storev
//      LLVM: %[[INIT:.*]] = call i32 @_Z11produce_intv()
// LLVM-NEXT: store i32 %[[INIT]], ptr %[[SLOT:.*]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT: %{{.+}} = load i32, ptr %[[SLOT]], align 4, !invariant.group !{{.+}}
//      LLVM: ret i32
// LLVM-NEXT: }

// OGCG-LABEL: @_Z22local_const_load_storev
//      OGCG: %[[INIT:.*]] = call {{.*}}i32 @_Z11produce_intv()
// OGCG-NEXT: store i32 %[[INIT]], ptr %[[SLOT:.*]], align 4
// OGCG-NEXT: call void @llvm.lifetime.start
// OGCG-NEXT: %{{.+}} = load i32, ptr %[[SLOT]], align 4
//      OGCG: ret i32
// OGCG-NEXT: }

int local_const_optimize() {
  const int x = produce_int();
  blackbox(x);
  blackbox(x);
  return x;
}

// CIR-LABEL: @_Z20local_const_optimizev()
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT: cir.call @_Z11produce_intv()
// CIR-NEXT: cir.store
// CIR-NEXT: cir.call @_Z8blackboxRKi(
// CIR-NEXT: cir.call @_Z8blackboxRKi(
// CIR-NEXT: cir.load
// CIR-NEXT: cir.store
// CIR-NEXT: cir.load
// CIR-NEXT: cir.return

// CIR-OPT-LABEL: @_Z20local_const_optimizev()
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-OPT-NEXT: cir.call @_Z11produce_intv()
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.call @_Z8blackboxRKi(
// CIR-OPT-NEXT: cir.call @_Z8blackboxRKi(
// CIR-OPT-NEXT: cir.load
// CIR-OPT-NEXT: cir.store
// CIR-OPT-NEXT: cir.load
// CIR-OPT-NEXT: cir.return

// LLVM-LABEL: @_Z20local_const_optimizev()
// LLVM-NEXT:    %[[RET:.*]] = alloca i32
// LLVM-NEXT:    %[[X:.*]] = alloca i32
// LLVM-NEXT:    %[[INIT:.*]] = call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[INIT]], ptr %[[X]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr %[[X]])
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr %[[X]])
// LLVM-NEXT:     %[[LOAD:.*]] = load i32, ptr %[[X]], align 4, !invariant.group !{{.*}}
// LLVM-NEXT:    store i32 %[[LOAD]], ptr %[[RET]]
// LLVM-NEXT:    %[[RET_LOAD:.*]] = load i32, ptr %[[RET]]
// LLVM-NEXT:    ret i32 %[[RET_LOAD]]
// LLVM-NEXT:  }

// OGCG-LABEL: @_Z20local_const_optimizev()
// OGCG-NEXT:  entry:
// OGCG-NEXT:    %[[X:.*]] = alloca i32
// OGCG-NEXT:    call void @llvm.lifetime.start
// OGCG-NEXT:    %[[INIT:.*]] = tail call {{.*}}i32 @_Z11produce_intv()
// OGCG-NEXT:    store i32 %[[INIT]], ptr %[[X]], align 4
// OGCG-NEXT:    call void @_Z8blackboxRKi(ptr {{.*}}%[[X]])
// OGCG-NEXT:    call void @_Z8blackboxRKi(ptr {{.*}}%[[X]])
// OGCG-NEXT:    %[[RET_LOAD:.*]] = load i32, ptr %[[X]]
// OGCG-NEXT:    call void @llvm.lifetime.end
// OGCG-NEXT:    ret i32 %[[RET_LOAD]]
// OGCG-NEXT:  }

int local_scoped_const() {
  {
    const int x = produce_int();
    blackbox(x);
    return x;
  }
}

// CIR-LABEL: @_Z18local_scoped_constv()
// CIR-NEXT:    cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-NEXT:    cir.scope {
// CIR-NEXT:      %[[X_SLOT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT:      %[[INIT:.*]] = cir.call @_Z11produce_intv() : () -> !s32i
// CIR-NEXT:      cir.store{{.*}} %[[INIT]], %[[X_SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      cir.call @_Z8blackboxRKi(%[[X_SLOT]]) : (!cir.ptr<!s32i>) -> ()
// CIR-NEXT:      %[[X_RELOAD:.*]] = cir.load{{.*}} %[[X_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:      cir.store{{.*}} %[[X_RELOAD]], %[[RET_SLOT:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:      %[[RET:.*]] = cir.load{{.*}} %[[RET_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:      cir.return %[[RET]] : !s32i
// CIR-NEXT:    }
// CIR-NEXT:  cir.trap
// CIR-NEXT:  }

// CIR-OPT-LABEL: @_Z18local_scoped_constv()
// CIR-OPT-NEXT:    %[[X_SLOT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-OPT-NEXT:    cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR-OPT-NEXT:    cir.scope {
// CIR-OPT-NEXT:      %[[INIT:.*]] = cir.call @_Z11produce_intv() : () -> !s32i
// CIR-OPT-NEXT:      cir.store{{.*}} %[[INIT]], %[[X_SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-OPT-NEXT:      cir.call @_Z8blackboxRKi(%[[X_SLOT]]) : (!cir.ptr<!s32i>) -> ()
// CIR-OPT-NEXT:      %[[X_RELOAD:.*]] = cir.load{{.*}} %[[X_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR-OPT-NEXT:      cir.store{{.*}} %[[X_RELOAD]], %[[RET_SLOT:.*]] : !s32i, !cir.ptr<!s32i>
// CIR-OPT-NEXT:      %[[RET:.*]] = cir.load{{.*}} %[[RET_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR-OPT-NEXT:      cir.return %[[RET]] : !s32i
// CIR-OPT-NEXT:    }
// CIR-OPT-NEXT:  cir.unreachable
// CIR-OPT-NEXT:  }

// LLVM-LABEL: @_Z18local_scoped_constv()
// LLVM-NEXT:    %[[X:.*]] = alloca i32
// LLVM-NEXT:    %[[RET:.*]] = alloca i32
// LLVM-NEXT:    br label %[[BODY:.*]]
//      LLVM:    [[BODY]]:
// LLVM-NEXT:    %[[INIT:.*]] = call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[INIT]], ptr %[[X]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr %[[X]])
// LLVM-NEXT:    %[[X_LOAD:.*]] = load i32, ptr %[[X]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT:    store i32 %[[X_LOAD]], ptr %[[RET]], align 4
// LLVM-NEXT:    %[[RET_LOAD:.*]] = load i32, ptr %[[RET]]
// LLVM-NEXT:    ret i32 %[[RET_LOAD]]
//      LLVM:    unreachable
// LLVM-NEXT:  }

// OGCG-LABEL: @_Z18local_scoped_constv()
// OGCG-NEXT:  entry:
// OGCG-NEXT:    %[[X:.*]] = alloca i32
// OGCG-NEXT:    call void @llvm.lifetime.start
// OGCG-NEXT:    %[[INIT:.*]] = tail call {{.*}}i32 @_Z11produce_intv()
// OGCG-NEXT:    store i32 %[[INIT]], ptr %[[X]], align 4
// OGCG-NEXT:    call void @_Z8blackboxRKi(ptr {{.*}}%[[X]])
// OGCG-NEXT:    %[[X_LOAD:.*]] = load i32, ptr %[[X]], align 4
// OGCG-NEXT:     call void @llvm.lifetime.end
// OGCG-NEXT:    ret i32 %[[X_LOAD]]
// OGCG-NEXT:  }

void local_const_in_loop() {
  for (int i = 0; i < 10; ++i) {
    const int x = produce_int();
    blackbox(x);
    consume(x);
  }
}

// CIR-LABEL: @_Z19local_const_in_loopv
// CIR-NEXT:    cir.scope {
// CIR-NEXT:      cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR-NEXT:      cir.const
// CIR-NEXT:      cir.store
// CIR-NEXT:      cir.for : cond {
//      CIR:      } body {
// CIR-NEXT:        cir.scope {
// CIR-NEXT:          %[[X_SLOT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT:          %[[INIT:.*]] = cir.call @_Z11produce_intv() : () -> !s32i
// CIR-NEXT:          cir.store{{.*}} %[[INIT]], %[[X_SLOT]] : !s32i, !cir.ptr<!s32i>
// CIR-NEXT:          cir.call @_Z8blackboxRKi(%[[X_SLOT]]) : (!cir.ptr<!s32i>) -> ()
// CIR-NEXT:          %[[X_RELOAD:.*]] = cir.load{{.*}} %[[X_SLOT]] : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:          cir.call @_Z7consumei(%[[X_RELOAD]]) : (!s32i) -> ()
// CIR-NEXT:        }
// CIR-NEXT:        cir.yield
// CIR-NEXT:      } step {
//      CIR:      }
// CIR-NEXT:    }
// CIR-NEXT:    cir.return
// CIR-NEXT:  }

// CIR-OPT-LABEL: @_Z19local_const_in_loopv
// CIR-OPT-NEXT:    cir.alloca !s32i, !cir.ptr<!s32i>, ["i", init]
// CIR-OPT-NEXT:    %[[X_SLOT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-OPT-NEXT:    cir.scope {
// CIR-OPT-NEXT:      cir.const
// CIR-OPT-NEXT:      cir.store
// CIR-OPT-NEXT:      cir.for : cond {
//      CIR-OPT:      } body {
// CIR-OPT-NEXT:        cir.scope {
// CIR-OPT-NEXT:          %[[INVARIANT:.*]] = cir.invariant_group %[[X_SLOT]] : !cir.ptr<!s32i>
// CIR-OPT-NEXT:          %[[INIT:.*]] = cir.call @_Z11produce_intv() : () -> !s32i
// CIR-OPT-NEXT:          cir.store{{.*}} %[[INIT]], %[[INVARIANT]] : !s32i, !cir.ptr<!s32i>
// CIR-OPT-NEXT:          cir.call @_Z8blackboxRKi(%[[INVARIANT]]) : (!cir.ptr<!s32i>) -> ()
// CIR-OPT-NEXT:          %[[X_RELOAD:.*]] = cir.load{{.*}} %[[INVARIANT]] : !cir.ptr<!s32i>, !s32i
// CIR-OPT-NEXT:          cir.call @_Z7consumei(%[[X_RELOAD]]) : (!s32i) -> ()
// CIR-OPT-NEXT:        }
// CIR-OPT-NEXT:        cir.yield
// CIR-OPT-NEXT:      } step {
//      CIR-OPT:      }
// CIR-OPT-NEXT:    }
// CIR-OPT-NEXT:    cir.return
// CIR-OPT-NEXT:  }

// LLVM-LABEL: @_Z19local_const_in_loopv()
// LLVM-NEXT:    %[[I:.*]] = alloca i32
// LLVM-NEXT:    %[[X:.*]] = alloca i32
//      LLVM:    %[[X_PTR:.*]] = call ptr @llvm.launder.invariant.group.p0(ptr %[[X]])
// LLVM-NEXT:    %[[INIT:.*]] = call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[INIT]], ptr %[[X_PTR]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT:    call void @_Z8blackboxRKi(ptr %[[X_PTR]])
// LLVM-NEXT:    %[[X_PTR_LOAD:.*]] = load i32, ptr %[[X_PTR]], align 4, !invariant.group !{{.+}}
// LLVM-NEXT:    call void @_Z7consumei(i32 %[[X_PTR_LOAD]])
//      LLVM:  }

// OGCG-LABEL: @_Z19local_const_in_loopv()
// OGCG-NEXT:  entry:
// OGCG-NEXT:    %[[X:.*]] = alloca i32
// OGCG-NEXT:    br label %[[BODY:.*]]
//      OGCG:    [[BODY]]:
// OGCG-NEXT:    %[[PHI:.*]] = phi i32 [ 0, %entry ], [ %inc, %[[BODY]] ]
// OGCG-NEXT:    call void @llvm.lifetime.start
// OGCG-NEXT:    %[[INIT:.*]] = call {{.*}}i32 @_Z11produce_intv()
// OGCG-NEXT:    store i32 %[[INIT]], ptr %[[X]], align 4
// OGCG-NEXT:    call void @_Z8blackboxRKi(ptr {{.*}}%[[X]])
// OGCG-NEXT:    %[[X_LOAD:.*]] = load i32, ptr %[[X]], align 4
// OGCG-NEXT:    call void @_Z7consumei(i32 {{.*}}%[[X_LOAD]])
// OGCG-NEXT:    call void @llvm.lifetime.end
//      OGCG:  }

void local_const_in_while_condition() {
  while (const int x = produce_int()) {
    blackbox(x);
  }
}
// CIR-LABEL: @_Z30local_const_in_while_conditionv
// CIR-NEXT: cir.scope {
// CIR-NEXT:   cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init, const]
// CIR-NEXT:   cir.while {
// CIR-NEXT:     cir.call @_Z11produce_intv
// CIR-NEXT:     cir.store
// CIR-NEXT:     cir.load
// CIR-NEXT:     cir.cast
// CIR-NEXT:     cir.condition
// CIR-NEXT:   } do {
// CIR-NEXT:     cir.scope {
// CIR-NEXT:       cir.call @_Z8blackboxRKi(
// CIR-NEXT:     }
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }
// CIR-NEXT: }
// CIR-NEXT: cir.return

// CIR-OPT-LABEL: @_Z30local_const_in_while_conditionv
// CIR-OPT-NEXT: cir.alloca !s32i, !cir.ptr<!s32i>, ["x", init]
// CIR-OPT-NEXT: cir.scope {
// CIR-OPT-NEXT:   cir.while {
// CIR-OPT-NEXT:     cir.call @_Z11produce_intv
// CIR-OPT-NEXT:     cir.store
// CIR-OPT-NEXT:     cir.load
// CIR-OPT-NEXT:     cir.cast
// CIR-OPT-NEXT:     cir.condition
// CIR-OPT-NEXT:   } do {
// CIR-OPT-NEXT:     cir.scope {
// CIR-OPT-NEXT:       cir.call @_Z8blackboxRKi(
// CIR-OPT-NEXT:     }
// CIR-OPT-NEXT:     cir.yield
// CIR-OPT-NEXT:   }
// CIR-OPT-NEXT: }
// CIR-OPT-NEXT: cir.return

// LLVM-LABEL: @_Z30local_const_in_while_conditionv()
//      LLVM:    %[[X_SLOT:.*]] = alloca i32
//      LLVM:    %[[INIT:.*]] = call i32 @_Z11produce_intv()
// LLVM-NEXT:    store i32 %[[INIT]], ptr %[[X_SLOT]], align 4
// LLVM-NEXT:    %[[LOAD:.*]] = load i32, ptr %[[X_SLOT]]
// LLVM-NEXT:    %[[LOOP_COND:.+]] = icmp ne i32 %[[LOAD]], 0
// LLVM-NEXT:    br i1 %[[LOOP_COND]], label %[[LOOP_BODY:.+]], label %{{.+}}
//      LLVM:  [[LOOP_BODY]]:
//      LLVM:    call void @_Z8blackboxRKi(ptr %[[X_SLOT]])
//      LLVM:  ret void
// LLVM-NET:  }

// OGCG-LABEL: @_Z30local_const_in_while_conditionv()
//      OGCG:    %[[X_SLOT:.*]] = alloca i32
// OGCG-NEXT:    call void @llvm.lifetime.start
//      OGCG:    %[[INIT:.*]] = tail call {{.*}}i32 @_Z11produce_intv()
// OGCG-NEXT:    store i32 %[[INIT]], ptr %[[X_SLOT]], align 4
// OGCG-NEXT:    %[[LOOP_COND:.*]] = icmp eq i32 %[[INIT]], 0
// OGCG-NEXT:    br i1 %[[LOOP_COND]], label %{{.+}}, label %[[LOOP_BODY:.+]]
//      OGCG:  [[LOOP_BODY]]:
//      OGCG:    call void @_Z8blackboxRKi(ptr {{.*}}%[[X_SLOT]])
// OGCG-NEXT:    call void @llvm.lifetime.end
// OGCG-NEXT:    call void @llvm.lifetime.start
//      OGCG:    %[[INIT:.*]] = call {{.*}}i32 @_Z11produce_intv()
// OGCG-NEXT:    store i32 %[[INIT]], ptr %[[X_SLOT]], align 4
// OGCG-NEXT:    %[[LOOP_COND:.*]] = icmp eq i32 %[[INIT]], 0
// OGCG-NEXT:    br i1 %[[LOOP_COND]], label %{{.+}}, label %[[LOOP_BODY]]
//      OGCG:  ret void
//      OGCG:  }
