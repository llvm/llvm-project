// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-o0.cir
// RUN: FileCheck --input-file=%t-o0.cir %s --implicit-check-not "cir.lifetime"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s --check-prefix=CIR-EH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fcxx-exceptions -fexceptions -fclangir -emit-llvm -disable-llvm-passes %s -o %t-eh.ll
// RUN: FileCheck --input-file=%t-eh.ll %s --check-prefixes=LLVM-EH

void use(int);

// A scalar automatic variable gets a lifetime.start at its declaration and a
// matching lifetime.end when its scope is left.
void f() {
  int x;
  use(x);
}

// CIR-LABEL: cir.func{{.*}} @_Z1fv()
// CIR:         %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!s32i>
// CIR:         cir.lifetime.start %[[X]] : !cir.ptr<!s32i>
// CIR:         cir.cleanup.scope {
// CIR:           cir.call @_Z3usei
// CIR:         } cleanup normal {
// CIR:           cir.lifetime.end %[[X]] : !cir.ptr<!s32i>
// CIR:         }

// LLVM-LABEL: define{{.*}} void @_Z1fv()
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[X]])
// LLVM:         call void @_Z3usei
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[X]])

struct S {
  ~S();
};

// The destructor runs before lifetime.end: the end marker is the outermost
// cleanup, so it is emitted after the destructor call. FileCheck matches in
// order, which pins the relative ordering.
void g() {
  S s;
}

// CIR-LABEL: cir.func{{.*}} @_Z1gv()
// CIR:         %[[S:.*]] = cir.alloca "s" {{.*}} : !cir.ptr<!rec_S>
// CIR:         cir.lifetime.start %[[S]] : !cir.ptr<!rec_S>
// CIR:         cir.call @_ZN1SD1Ev(%[[S]])
// CIR:         cir.lifetime.end %[[S]] : !cir.ptr<!rec_S>

// LLVM-LABEL: define{{.*}} void @_Z1gv()
// LLVM:         %[[S:.*]] = alloca %struct.S
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[S]])
// LLVM:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[S]])

// A statement that can bypass a local's initialization -- switch, label, or
// indirect goto -- miscompiles under stack coloring (PR28267). Lacking classic
// CodeGen's per-decl bypass analysis, we conservatively drop lifetime markers
// for the *whole* function whenever any such statement is present, even at -O2
// and even for locals (like `x` below) that are not themselves bypassed.

void bypass_switch(int n) {
  int x;
  use(x);
  switch (n) {
  case 0:
    return;
  }
}

// CIR-LABEL: cir.func{{.*}}bypass_switch
// CIR-NOT:     cir.lifetime

// LLVM-LABEL: define{{.*}}bypass_switch
// LLVM-NOT:    call void @llvm.lifetime

void bypass_label(int n) {
  int x;
  use(x);
target:
  if (n)
    goto target;
}

// CIR-LABEL: cir.func{{.*}}bypass_label
// CIR-NOT:     cir.lifetime

void bypass_indirect_goto() {
  int x;
  use(x);
  void *p = &&target;
  goto *p;
target:
  return;
}

// CIR-LABEL: cir.func{{.*}}bypass_indirect_goto
// CIR-NOT:     cir.lifetime

// A local declared inside the body region of an if statement is scoped to that
// region: its lifetime.start/end are nested in the region and the end marker
// is the region's cleanup, not the function's.
void if_body(int n) {
  if (n) {
    int x;
    use(x);
  }
  use(n);
}

// CIR-LABEL: cir.func{{.*}} @_Z7if_bodyi
// CIR:         cir.if %{{.*}} {
// CIR:           %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!s32i>
// CIR:           cir.lifetime.start %[[X]] : !cir.ptr<!s32i>
// CIR:           cir.cleanup.scope {
// CIR:             cir.call @_Z3usei
// CIR:           } cleanup normal {
// CIR:             cir.lifetime.end %[[X]] : !cir.ptr<!s32i>
// CIR-NEXT:        cir.yield
// CIR-NEXT:      }
// CIR-NEXT:    }
// CIR:         cir.call @_Z3usei

// LLVM-LABEL: define{{.*}} void @_Z7if_bodyi
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         br i1 %{{.*}}, label %[[IF_BODY:[0-9]+]], label %[[IF_END:[0-9]+]]
// LLVM:       [[IF_BODY]]:
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[X]])
// LLVM:         call void @_Z3usei
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[X]])
// LLVM:       [[IF_END]]:
// LLVM:         call void @_Z3usei

// With exceptions enabled the scope cleanup runs on both the normal and the
// exceptional edge, so the cleanup kind is "all" and lifetime.end is emitted in
// the EH cleanup handler (the landing pad) as well as on the normal path. The
// may_throw() call is what forces an unwind edge.
void may_throw();

void eh_cleanup() {
  int x;
  may_throw();
  use(x);
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z10eh_cleanupv
// CIR-EH:         %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!s32i>
// CIR-EH:         cir.lifetime.start %[[X]] : !cir.ptr<!s32i>
// CIR-EH:         cir.cleanup.scope {
// CIR-EH:           cir.call @_Z9may_throwv()
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.lifetime.end %[[X]] : !cir.ptr<!s32i>
// CIR-EH:         }

// LLVM-EH-LABEL: define{{.*}} void @_Z10eh_cleanupv()
// LLVM-EH:         %[[X:.*]] = alloca i32
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[X]])
// LLVM-EH:         invoke void @_Z9may_throwv()
// The normal-path end marker.
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[X]])
// The EH cleanup handler runs the same end marker on the unwind path.
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH-NEXT:      cleanup
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[X]])

// A loop condition variable is destroyed and re-created on every iteration
// (C++ [stmt.while]p2). Its lifetime starts in the condition region and ends in
// the loop cleanup region, which runs on both the back edge and the exit edge.
int source();

void while_condvar() {
  while (int c = source())
    use(c);
}

// CIR-LABEL: cir.func{{.*}} @_Z13while_condvarv
// CIR:           %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!s32i>
// CIR:           cir.while {
// CIR:             cir.lifetime.start %[[C]] : !cir.ptr<!s32i>
// CIR:           } do {
// CIR:           } cleanup normal {
// CIR:             cir.lifetime.end %[[C]] : !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} void @_Z13while_condvarv
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[C]])

// CIR-EH-LABEL: cir.func{{.*}} @_Z13while_condvarv
// CIR-EH:         %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!s32i>
// CIR-EH:         cir.while {
// CIR-EH:           cir.lifetime.start %[[C]] : !cir.ptr<!s32i>
// CIR-EH:         } do {
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.lifetime.end %[[C]] : !cir.ptr<!s32i>

// LLVM-EH-LABEL: define{{.*}} void @_Z13while_condvarv
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH-NEXT:      cleanup
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])

void for_condvar() {
  for (; int c = source();)
    use(c);
}

// CIR-LABEL: cir.func{{.*}} @_Z11for_condvarv
// CIR:           %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!s32i>
// CIR:           cir.for : cond {
// CIR:             cir.lifetime.start %[[C]] : !cir.ptr<!s32i>
// CIR:           } body {
// CIR:           } step {
// CIR:           } cleanup normal {
// CIR:             cir.lifetime.end %[[C]] : !cir.ptr<!s32i>

// LLVM-LABEL: define{{.*}} void @_Z11for_condvarv
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[C]])

// CIR-EH-LABEL: cir.func{{.*}} @_Z11for_condvarv
// CIR-EH:         %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!s32i>
// CIR-EH:         cir.for : cond {
// CIR-EH:           cir.lifetime.start %[[C]] : !cir.ptr<!s32i>
// CIR-EH:         } body {
// CIR-EH:         } step {
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.lifetime.end %[[C]] : !cir.ptr<!s32i>

// LLVM-EH-LABEL: define{{.*}} void @_Z11for_condvarv
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH-NEXT:      cleanup
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])

struct LoopCond {
  operator bool() const;
  ~LoopCond();
};

// A non-trivial condition variable runs its destructor before lifetime.end in
// the loop cleanup region.
void while_record_condvar() {
  while (LoopCond c{}) {}
}

// CIR-LABEL: cir.func{{.*}} @_Z20while_record_condvarv
// CIR:           %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!rec_LoopCond>
// CIR:           cir.while {
// CIR:             cir.lifetime.start %[[C]] : !cir.ptr<!rec_LoopCond>
// CIR:           } do {
// CIR:           } cleanup normal {
// CIR:             cir.call @_ZN8LoopCondD1Ev(%[[C]])
// CIR:             cir.lifetime.end %[[C]] : !cir.ptr<!rec_LoopCond>

// LLVM-LABEL: define{{.*}} void @_Z20while_record_condvarv
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM:         call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[C]])
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[C]])

// CIR-EH-LABEL: cir.func{{.*}} @_Z20while_record_condvarv
// CIR-EH:         %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!rec_LoopCond>
// CIR-EH:         cir.while {
// CIR-EH:           cir.lifetime.start %[[C]] : !cir.ptr<!rec_LoopCond>
// CIR-EH:         } do {
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.call @_ZN8LoopCondD1Ev(%[[C]])
// CIR-EH:           cir.lifetime.end %[[C]] : !cir.ptr<!rec_LoopCond>

// LLVM-EH-LABEL: define{{.*}} void @_Z20while_record_condvarv
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[C]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH-NEXT:      cleanup
// LLVM-EH:         call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[C]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])

#ifdef __EXCEPTIONS

struct Ex {};

void catch_by_ref() {
  try {
    may_throw();
  } catch (const Ex &e) {
  }
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z12catch_by_refv
// CIR-EH:         %[[E:.*]] = cir.alloca "e" {{.*}} : !cir.ptr<!cir.ptr<!rec_Ex>>
// CIR-EH:         } catch [type #cir.global_view<@_ZTI2Ex>{{.*}}] (%[[TOK:[^:]*]]:
// CIR-EH-NEXT:      cir.lifetime.start %[[E]] : !cir.ptr<!cir.ptr<!rec_Ex>>
// CIR-EH-NEXT:      cir.cleanup.scope {
// CIR-EH-NEXT:        %[[CATCH_TOK:.*]], %{{.*}} = cir.begin_catch %[[TOK]]
// CIR-EH:             } cleanup all {
// CIR-EH:               cir.end_catch %[[CATCH_TOK]]
// CIR-EH:             }
// CIR-EH:           } cleanup all {
// CIR-EH-NEXT:        cir.lifetime.end %[[E]] : !cir.ptr<!cir.ptr<!rec_Ex>>

// LLVM-EH-LABEL: define{{.*}} void @_Z12catch_by_refv()
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[E:.*]])
// LLVM-EH:         call ptr @__cxa_begin_catch
// LLVM-EH:         call void @__cxa_end_catch()
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[E]])

struct Copy {
  Copy(const Copy &);
  ~Copy();
};

void catch_by_value() {
  try {
    may_throw();
  } catch (Copy c) {
  }
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z14catch_by_valuev
// CIR-EH:         %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!rec_Copy>
// CIR-EH:         } catch [type #cir.global_view<@_ZTI4Copy>{{.*}}] (%[[TOK:[^:]*]]:
// CIR-EH-NEXT:      cir.lifetime.start %[[C]] : !cir.ptr<!rec_Copy>
// CIR-EH-NEXT:      cir.cleanup.scope {
// CIR-EH-NEXT:        cir.construct_catch_param non_trivial_copy %[[TOK]] to %[[C]]
// CIR-EH-NEXT:        %[[CATCH_TOK:.*]], %{{.*}} = cir.begin_catch %[[TOK]]
// CIR-EH:                 cir.call @_ZN4CopyD1Ev(%[[C]])
// CIR-EH:               cir.end_catch %[[CATCH_TOK]]
// CIR-EH:           } cleanup all {
// CIR-EH-NEXT:        cir.lifetime.end %[[C]] : !cir.ptr<!rec_Copy>

// LLVM-EH-LABEL: define{{.*}} void @_Z14catch_by_valuev()
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         call ptr @__cxa_get_exception_ptr
// LLVM-EH:         call ptr @__cxa_begin_catch
// LLVM-EH:         call void @_ZN4CopyD1Ev(ptr {{.*}} %[[C]])
// LLVM-EH:         call void @__cxa_end_catch()
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])

#endif // __EXCEPTIONS
