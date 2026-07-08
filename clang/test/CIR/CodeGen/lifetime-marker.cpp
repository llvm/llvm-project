// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-o0.cir
// RUN: FileCheck --input-file=%t-o0.cir %s --check-prefix=O0
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s --check-prefix=CIR-EH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fcxx-exceptions -fexceptions -fclangir -emit-llvm -disable-llvm-passes %s -o %t-eh.ll
// RUN: FileCheck --input-file=%t-eh.ll %s --check-prefix=LLVM-EH

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
// CIR:         } cleanup normal {
// CIR:           cir.lifetime.end %[[X]] : !cir.ptr<!s32i>
// CIR:         }

// LLVM-LABEL: define{{.*}} void @_Z1fv()
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[X]])
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[X]])

// Without optimization no lifetime markers are emitted. Checked per function so
// a regression in a single function can't hide behind a passing global check.
// O0-LABEL: cir.func{{.*}} @_Z1fv()
// O0-NOT:     cir.lifetime

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

// O0-LABEL: cir.func{{.*}} @_Z1gv()
// O0-NOT:     cir.lifetime

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

// O0-LABEL: cir.func{{.*}}bypass_switch
// O0-NOT:     cir.lifetime

void bypass_label(int n) {
  int x;
  use(x);
target:
  if (n)
    goto target;
}

// CIR-LABEL: cir.func{{.*}}bypass_label
// CIR-NOT:     cir.lifetime

// O0-LABEL: cir.func{{.*}}bypass_label
// O0-NOT:     cir.lifetime

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

// O0-LABEL: cir.func{{.*}}bypass_indirect_goto
// O0-NOT:     cir.lifetime

// A local declared inside the body region of an if statement is scoped to that
// region: its lifetime.start/end are nested in the region and the end marker
// is the region's cleanup, not the function's.
void if_body(int n) {
  if (n) {
    int x;
    use(x);
  }
}

// CIR-LABEL: cir.func{{.*}} @_Z7if_bodyi
// CIR:         cir.if
// CIR:           %[[X:.*]] = cir.alloca "x" {{.*}} : !cir.ptr<!s32i>
// CIR:           cir.lifetime.start %[[X]] : !cir.ptr<!s32i>
// CIR:           cir.cleanup.scope {
// CIR:           } cleanup normal {
// CIR:             cir.lifetime.end %[[X]] : !cir.ptr<!s32i>
// CIR:           }

// LLVM-LABEL: define{{.*}} void @_Z7if_bodyi
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[X]])
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[X]])

// O0-LABEL: cir.func{{.*}} @_Z7if_bodyi
// O0-NOT:     cir.lifetime

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
// (C++ [stmt.while]p2), so its lifetime.end would have to run on both the
// loop's back edge and its exit edge. The structured cond region cannot
// express a cleanup on both edges (a cleanup scope would trap the loop's
// condition terminator inside it), so lifetime markers are suppressed for
// variables declared in a while or for condition.
int source();

void while_condvar() {
  while (int c = source())
    use(c);
}

// CIR-LABEL: cir.func{{.*}}while_condvar
// CIR-NOT:     cir.lifetime

// LLVM-LABEL: define{{.*}}while_condvar
// LLVM-NOT:    call void @llvm.lifetime

// O0-LABEL: cir.func{{.*}}while_condvar
// O0-NOT:     cir.lifetime

// CIR-EH-LABEL: cir.func{{.*}}while_condvar
// CIR-EH-NOT:     cir.lifetime

void for_condvar() {
  for (; int c = source();)
    use(c);
}

// CIR-LABEL: cir.func{{.*}}for_condvar
// CIR-NOT:     cir.lifetime

// O0-LABEL: cir.func{{.*}}for_condvar
// O0-NOT:     cir.lifetime
