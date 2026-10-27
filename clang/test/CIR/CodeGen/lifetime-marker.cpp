// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefixes=CIR,EXT-CIR,EXT-CIR-NOEH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O2 -fclangir -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,EXT-LLVM,EXT-LLVM-NOEH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t-o0.cir
// RUN: FileCheck --input-file=%t-o0.cir %s --implicit-check-not "cir.lifetime"
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fcxx-exceptions -fexceptions -fclangir -emit-cir %s -o %t-eh.cir
// RUN: FileCheck --input-file=%t-eh.cir %s --check-prefixes=CIR-EH,EXT-CIR,EXT-CIR-EH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fcxx-exceptions -fexceptions -fclangir -emit-llvm -disable-llvm-passes %s -o %t-eh.ll
// RUN: FileCheck --input-file=%t-eh.ll %s --check-prefixes=LLVM-EH,EXT-LLVM,EXT-LLVM-EH
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -fsanitize-address-use-after-scope -fclangir -emit-cir %s -o %t-precise.cir
// RUN: FileCheck --input-file=%t-precise.cir %s --check-prefix=PRECISE
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -O1 -disable-lifetime-markers -fclangir -emit-cir %s -o %t-disabled.cir
// RUN: FileCheck --input-file=%t-disabled.cir %s --implicit-check-not "cir.lifetime"

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
// LLVM:         %[[N:.*]] = alloca i32
// LLVM:         %[[X:.*]] = alloca i32
// LLVM:         br i1 %{{.*}}, label %[[IF_BODY:[0-9]+]], label %[[IF_END:[0-9]+]]
// LLVM:       [[IF_BODY]]:
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[X]])
// LLVM:         call void @_Z3usei
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[X]])
// LLVM:       [[IF_END]]:
// LLVM:         call void @_Z3usei

// Lifetime cleanups retain the "all" kind, but must not introduce an unwind
// edge on their own. Only the normal-path marker is needed here.
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
// LLVM-EH-NOT:     personality
// LLVM-EH:         %[[X:.*]] = alloca i32
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[X]])
// LLVM-EH:         call void @_Z9may_throwv()
// LLVM-EH:         call void @_Z3usei
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

// LLVM-EH-LABEL: define{{.*}} void @_Z13while_condvarv()
// LLVM-EH-NOT:     personality
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         call{{.*}} i32 @_Z6sourcev()
// LLVM-EH:         call void @_Z3usei
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

// LLVM-EH-LABEL: define{{.*}} void @_Z11for_condvarv()
// LLVM-EH-NOT:     personality
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         call{{.*}} i32 @_Z6sourcev()
// LLVM-EH:         call void @_Z3usei
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

LoopCond makeLoopCond();

// Destroy the extended temporary, then end its lifetime and the reference's.
void while_extended_condvar() {
  while (const LoopCond &c = makeLoopCond()) {}
}

// EXT-CIR-LABEL: cir.func{{.*}} @_Z22while_extended_condvarv
// EXT-CIR:           %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_LoopCond>
// EXT-CIR:           %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!cir.ptr<!rec_LoopCond>>
// EXT-CIR:           cir.while {
// EXT-CIR-NEXT:        cir.lifetime.start %[[C]] : !cir.ptr<!cir.ptr<!rec_LoopCond>>
// EXT-CIR:             cir.lifetime.start %[[TMP]] : !cir.ptr<!rec_LoopCond>
// EXT-CIR-NEXT:        cir.call @_Z12makeLoopCondv(%[[TMP]])
// EXT-CIR:           } do {
// EXT-CIR-NOEH:      } cleanup normal {
// EXT-CIR-EH:        } cleanup all {
// EXT-CIR:             cir.if %{{.*}} {
// EXT-CIR-NEXT:          cir.call @_ZN8LoopCondD1Ev(%[[TMP]]){{.*}}
// EXT-CIR-NEXT:        }
// EXT-CIR-NEXT:        cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_LoopCond>
// EXT-CIR-NEXT:        cir.lifetime.end %[[C]] : !cir.ptr<!cir.ptr<!rec_LoopCond>>
// EXT-CIR-NEXT:        cir.yield

// EXT-LLVM-LABEL: define{{.*}} void @_Z22while_extended_condvarv
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// EXT-LLVM-NOEH:    call void @_Z12makeLoopCondv(ptr {{.*}} %[[TMP]])
// EXT-LLVM-EH:      invoke void @_Z12makeLoopCondv(ptr {{.*}} %[[TMP]])
// EXT-LLVM:         call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[TMP]])
// EXT-LLVM:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// EXT-LLVM-NEXT:    call void @llvm.lifetime.end.p0(ptr %[[C]])
// EXT-LLVM-EH:      landingpad { ptr, i32 }
// EXT-LLVM-EH-NEXT:        cleanup
// EXT-LLVM-EH:      call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[TMP]])
// EXT-LLVM-EH:      call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// EXT-LLVM-EH-NEXT: call void @llvm.lifetime.end.p0(ptr %[[C]])
// EXT-LLVM-EH:      resume

struct LoopCondHolder {
  const LoopCond &c;
  ~LoopCondHolder();
  operator bool() const;
};

// Destroy the record, then its extended temporary, then end their lifetimes.
void while_extended_record_condvar() {
  while (LoopCondHolder h{makeLoopCond()}) {}
}

// EXT-CIR-LABEL: cir.func{{.*}} @_Z29while_extended_record_condvarv
// EXT-CIR:           %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_LoopCond>
// EXT-CIR:           %[[H:.*]] = cir.alloca "h" {{.*}} : !cir.ptr<!rec_LoopCondHolder>
// EXT-CIR:           cir.while {
// EXT-CIR-NEXT:        cir.lifetime.start %[[H]] : !cir.ptr<!rec_LoopCondHolder>
// EXT-CIR:             cir.lifetime.start %[[TMP]] : !cir.ptr<!rec_LoopCond>
// EXT-CIR-NEXT:        cir.call @_Z12makeLoopCondv(%[[TMP]])
// EXT-CIR-NOEH:      } cleanup normal {
// EXT-CIR-EH:        } cleanup all {
// EXT-CIR:             cir.if %{{.*}} {
// EXT-CIR-NEXT:          cir.call @_ZN14LoopCondHolderD1Ev(%[[H]]){{.*}}
// EXT-CIR-NEXT:        }
// EXT-CIR:             cir.if %{{.*}} {
// EXT-CIR-NEXT:          cir.call @_ZN8LoopCondD1Ev(%[[TMP]]){{.*}}
// EXT-CIR-NEXT:        }
// EXT-CIR-NEXT:        cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_LoopCond>
// EXT-CIR-NEXT:        cir.lifetime.end %[[H]] : !cir.ptr<!rec_LoopCondHolder>
// EXT-CIR-NEXT:        cir.yield

// EXT-LLVM-LABEL: define{{.*}} void @_Z29while_extended_record_condvarv
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[H:.*]])
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// EXT-LLVM:         call void @_ZN14LoopCondHolderD1Ev(ptr {{.*}} %[[H]])
// EXT-LLVM:         call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[TMP]])
// EXT-LLVM:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// EXT-LLVM-NEXT:    call void @llvm.lifetime.end.p0(ptr %[[H]])
// EXT-LLVM-EH:      landingpad { ptr, i32 }
// EXT-LLVM-EH-NEXT:        cleanup
// EXT-LLVM-EH:      call void @_ZN14LoopCondHolderD1Ev(ptr {{.*}} %[[H]])
// EXT-LLVM-EH:      call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[TMP]])
// EXT-LLVM-EH:      call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// EXT-LLVM-EH-NEXT: call void @llvm.lifetime.end.p0(ptr %[[H]])
// EXT-LLVM-EH:      resume

struct TrivialCond {
  int x;
  operator bool() const;
};

TrivialCond makeTrivialCond();

// A trivially destructible extended temporary only needs its lifetime ended
// in the loop cleanup region, so no active flag is created. A cleanup holding
// only lifetime markers needs no landing pad.
void while_extended_trivial_condvar() {
  while (const TrivialCond &c = makeTrivialCond()) {}
}

// EXT-CIR-LABEL: cir.func{{.*}} @_Z30while_extended_trivial_condvarv
// EXT-CIR-NOT:       cond.cleanup.isactive
// EXT-CIR:           %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_TrivialCond>
// EXT-CIR:           %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// EXT-CIR:           cir.while {
// EXT-CIR-NEXT:        cir.lifetime.start %[[C]] : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// EXT-CIR-NEXT:        cir.lifetime.start %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// EXT-CIR-NEXT:        cir.call @_Z15makeTrivialCondv()
// EXT-CIR:           } do {
// EXT-CIR-NOEH:      } cleanup normal {
// EXT-CIR-EH:        } cleanup all {
// EXT-CIR-NEXT:        cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// EXT-CIR-NEXT:        cir.lifetime.end %[[C]] : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// EXT-CIR-NEXT:        cir.yield

// EXT-LLVM-LABEL: define{{.*}} void @_Z30while_extended_trivial_condvarv()
// EXT-LLVM-NOT:     personality
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// EXT-LLVM-NEXT:    call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// EXT-LLVM-NEXT:    call i32 @_Z15makeTrivialCondv()
// EXT-LLVM:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// EXT-LLVM-NEXT:    call void @llvm.lifetime.end.p0(ptr %[[C]])

void for_extended_trivial_condvar() {
  for (; const TrivialCond &c = makeTrivialCond();) {}
}

// EXT-CIR-LABEL: cir.func{{.*}} @_Z28for_extended_trivial_condvarv
// EXT-CIR:           %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_TrivialCond>
// EXT-CIR:           %[[C:.*]] = cir.alloca "c" {{.*}} : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// EXT-CIR:           cir.for : cond {
// EXT-CIR-NEXT:        cir.lifetime.start %[[C]] : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// EXT-CIR-NEXT:        cir.lifetime.start %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// EXT-CIR:           } step {
// EXT-CIR-NOEH:      } cleanup normal {
// EXT-CIR-EH:        } cleanup all {
// EXT-CIR-NEXT:        cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// EXT-CIR-NEXT:        cir.lifetime.end %[[C]] : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// EXT-CIR-NEXT:        cir.yield

// EXT-LLVM-LABEL: define{{.*}} void @_Z28for_extended_trivial_condvarv()
// EXT-LLVM-NOT:     personality
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// EXT-LLVM-NEXT:    call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// EXT-LLVM:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// EXT-LLVM-NEXT:    call void @llvm.lifetime.end.p0(ptr %[[C]])

struct MixedHolder {
  const LoopCond &c;
  const TrivialCond &t;
  operator bool() const;
};

// Extended temporaries end in reverse construction order. Only the destructor
// is guarded by the active flag. If the second initializer throws, the first
// temporary is destroyed by its full-expression EH cleanup, and the flag keeps
// the loop cleanup from destroying it again.
void while_extended_mixed_condvar() {
  while (MixedHolder h{makeLoopCond(), makeTrivialCond()}) {}
}

// EXT-CIR-LABEL: cir.func{{.*}} @_Z28while_extended_mixed_condvarv
// EXT-CIR:           %[[CTMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_LoopCond>
// EXT-CIR:           %[[TTMP:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_TrivialCond>
// EXT-CIR:           %[[H:.*]] = cir.alloca "h" {{.*}} : !cir.ptr<!rec_MixedHolder>
// EXT-CIR:           cir.while {
// EXT-CIR-NEXT:        cir.lifetime.start %[[H]] : !cir.ptr<!rec_MixedHolder>
// EXT-CIR:             cir.lifetime.start %[[CTMP]] : !cir.ptr<!rec_LoopCond>
// EXT-CIR-NEXT:        cir.call @_Z12makeLoopCondv(%[[CTMP]])
// EXT-CIR:             cir.lifetime.start %[[TTMP]] : !cir.ptr<!rec_TrivialCond>
// EXT-CIR-NEXT:        cir.call @_Z15makeTrivialCondv()
// EXT-CIR-EH:          } cleanup eh {
// EXT-CIR-EH-NEXT:       cir.call @_ZN8LoopCondD1Ev(%[[CTMP]]){{.*}}
// EXT-CIR:           } do {
// EXT-CIR-NOEH:      } cleanup normal {
// EXT-CIR-EH:        } cleanup all {
// EXT-CIR-NEXT:        cir.lifetime.end %[[TTMP]] : !cir.ptr<!rec_TrivialCond>
// EXT-CIR:             cir.if %{{.*}} {
// EXT-CIR-NEXT:          cir.call @_ZN8LoopCondD1Ev(%[[CTMP]]){{.*}}
// EXT-CIR-NEXT:        }
// EXT-CIR-NEXT:        cir.lifetime.end %[[CTMP]] : !cir.ptr<!rec_LoopCond>
// EXT-CIR-NEXT:        cir.lifetime.end %[[H]] : !cir.ptr<!rec_MixedHolder>
// EXT-CIR-NEXT:        cir.yield

// EXT-LLVM-LABEL: define{{.*}} void @_Z28while_extended_mixed_condvarv
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[H:.*]])
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[CTMP:.*]])
// EXT-LLVM:         call void @llvm.lifetime.start.p0(ptr %[[TTMP:.*]])
// EXT-LLVM-NOEH:    call i32 @_Z15makeTrivialCondv()
// EXT-LLVM-EH:      invoke i32 @_Z15makeTrivialCondv()
// EXT-LLVM-EH:      landingpad { ptr, i32 }
// EXT-LLVM-EH-NEXT:        cleanup
// EXT-LLVM-EH:      call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[CTMP]])
// EXT-LLVM:         call void @llvm.lifetime.end.p0(ptr %[[TTMP]])
// EXT-LLVM:         call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[CTMP]])
// EXT-LLVM:         call void @llvm.lifetime.end.p0(ptr %[[CTMP]])
// EXT-LLVM-NEXT:    call void @llvm.lifetime.end.p0(ptr %[[H]])
// EXT-LLVM-EH:      landingpad { ptr, i32 }
// EXT-LLVM-EH-NEXT:        cleanup
// EXT-LLVM-EH:      call void @llvm.lifetime.end.p0(ptr %[[TTMP]])
// EXT-LLVM-EH:      call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[CTMP]])
// EXT-LLVM-EH:      call void @llvm.lifetime.end.p0(ptr %[[CTMP]])
// EXT-LLVM-EH-NEXT: call void @llvm.lifetime.end.p0(ptr %[[H]])
// EXT-LLVM-EH:      resume

int consume(const TrivialCond &);

// A full-expression temporary's lifetime ends with the full-expression.
void full_expr_temporary() {
  use(consume(makeTrivialCond()));
}

// CIR-LABEL: cir.func{{.*}} @_Z19full_expr_temporaryv
// CIR:         %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_TrivialCond>
// CIR:         cir.lifetime.start %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// CIR-NEXT:    cir.cleanup.scope {
// CIR:           cir.call @_Z7consumeRK11TrivialCond(%[[TMP]])
// CIR:           cir.call @_Z3usei
// CIR:         } cleanup normal {
// CIR-NEXT:      cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_TrivialCond>

// CIR-EH-LABEL: cir.func{{.*}} @_Z19full_expr_temporaryv
// CIR-EH:         %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_TrivialCond>
// CIR-EH:         cir.lifetime.start %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// CIR-EH-NEXT:    cir.cleanup.scope {
// CIR-EH:         } cleanup all {
// CIR-EH-NEXT:      cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_TrivialCond>

// LLVM-LABEL: define{{.*}} void @_Z19full_expr_temporaryv
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// LLVM:         call noundef i32 @_Z7consumeRK11TrivialCond(ptr {{.*}} %[[TMP]])
// LLVM:         call void @_Z3usei
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])

// LLVM-EH-LABEL: define{{.*}} void @_Z19full_expr_temporaryv()
// LLVM-EH-NOT:     personality
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// LLVM-EH:         call void @_Z3usei
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])

// A temporary extended by a local reference lives until the end of the
// enclosing scope, and its lifetime ends before the reference's.
void extended_temporary() {
  const TrivialCond &r = makeTrivialCond();
  use(r.x);
}

// CIR-LABEL: cir.func{{.*}} @_Z18extended_temporaryv
// CIR:         %[[TMP:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_TrivialCond>
// CIR:         %[[R:.*]] = cir.alloca "r" {{.*}} : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// CIR:         cir.lifetime.start %[[R]] : !cir.ptr<!cir.ptr<!rec_TrivialCond>>
// CIR:         cir.lifetime.start %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// CIR-NEXT:    cir.call @_Z15makeTrivialCondv()
// CIR:         cir.call @_Z3usei
// CIR:         } cleanup normal {
// CIR-NEXT:      cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_TrivialCond>
// CIR:         } cleanup normal {
// CIR-NEXT:      cir.lifetime.end %[[R]] : !cir.ptr<!cir.ptr<!rec_TrivialCond>>

// LLVM-LABEL: define{{.*}} void @_Z18extended_temporaryv
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[R:.*]])
// LLVM:         call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// LLVM:         call void @_Z3usei
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// LLVM:         call void @llvm.lifetime.end.p0(ptr %[[R]])

// LLVM-EH-LABEL: define{{.*}} void @_Z18extended_temporaryv()
// LLVM-EH-NOT:     personality
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[R:.*]])
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// LLVM-EH:         call void @_Z3usei
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[R]])

extern "C" void consume(const int &);
extern "C" void after();

struct Guard {
  Guard(int);
  ~Guard();
  int value;
};

extern "C" void use_guard(const Guard &);

// Trivial destruction must not suppress the lifetime end. End the storage
// lifetime at the full-expression boundary, before the next statement.
extern "C" void scalar() {
  consume(42);
  after();
}

// CIR-LABEL: cir.func{{.*}} @scalar()
// CIR: %[[TMP:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!s32i>
// CIR: cir.lifetime.start %[[TMP]]
// CIR: cir.store{{.*}}, %[[TMP]]
// CIR: cir.call @consume(%[[TMP]])
// CIR: cir.lifetime.end %[[TMP]]
// CIR: cir.call @after()
// LLVM-LABEL: define{{.*}} void @scalar()
// LLVM: %[[TMP:.*]] = alloca i32
// LLVM: call void @llvm.lifetime.start.p0(ptr %[[TMP]])
// LLVM: call void @consume(ptr{{.*}} %[[TMP]])
// LLVM: call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// LLVM: call void @after()
// A lifetime-only cleanup does not introduce an unwind edge on its own.
// LLVM-EH-LABEL: define{{.*}} void @scalar()
// LLVM-EH-NOT: invoke
// LLVM-EH: call void @consume(
// LLVM-EH: call void @llvm.lifetime.end.p0(
// LLVM-EH: call void @after()
// LLVM-EH-NOT: landingpad
// LLVM-EH: ret void

// Register the end before the destructor so the destructor runs first.
extern "C" void nontrivial() {
  use_guard(Guard(1));
  after();
}

// CIR-LABEL: cir.func{{.*}} @nontrivial()
// CIR: %[[TMP:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!rec_Guard>
// CIR: cir.lifetime.start %[[TMP]]
// CIR: cir.call @_ZN5GuardC1Ei(%[[TMP]]
// CIR: cir.call @use_guard(%[[TMP]])
// CIR: cir.call @_ZN5GuardD1Ev(%[[TMP]])
// CIR: cir.lifetime.end %[[TMP]]
// CIR: cir.call @after()
// LLVM-LABEL: define{{.*}} void @nontrivial()
// LLVM: %[[TMP:.*]] = alloca %struct.Guard
// LLVM: call void @llvm.lifetime.start.p0(ptr %[[TMP]])
// LLVM: call void @use_guard(ptr{{.*}} %[[TMP]])
// LLVM: call void @_ZN5GuardD1Ev(ptr{{.*}} %[[TMP]])
// LLVM: call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// LLVM: call void @after()

// The pending end is promoted after initialization and belongs to the
// enclosing scope, rather than the initialization full-expression.
extern "C" void extended() {
  const Guard &ref = Guard(2);
  use_guard(ref);
  after();
}

// CIR-LABEL: cir.func{{.*}} @extended()
// CIR: %[[TMP:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!rec_Guard>
// CIR: cir.lifetime.start %[[TMP]]
// CIR: cir.call @use_guard(
// CIR: cir.call @after()
// CIR: cir.call @_ZN5GuardD1Ev(%[[TMP]])
// CIR: cir.lifetime.end %[[TMP]]
// LLVM-LABEL: define{{.*}} void @extended()
// LLVM: %[[TMP:.*]] = alloca %struct.Guard
// LLVM: call void @use_guard(
// LLVM: call void @after()
// LLVM: call void @_ZN5GuardD1Ev(ptr{{.*}} %[[TMP]])
// LLVM: call void @llvm.lifetime.end.p0(ptr %[[TMP]])

// The conditional pending entry contains only an address and a marker kind.
// Its empty type and null destroyer must never reach emitDestroy. Hoisting
// the start lets the full-expression cleanup end its lifetime without a flag.
extern "C" void conditional_scalar(bool enabled) {
  (void)(enabled && (consume(43), true));
  after();
}

// CIR-LABEL: cir.func{{.*}} @conditional_scalar(
// CIR: %[[TMP:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!s32i>
// CIR-NOT: cleanup.cond
// CIR: cir.lifetime.start %[[TMP]]
// CIR-NEXT: {{.*}}cir.ternary
// CIR: cir.call @consume(%[[TMP]])
// CIR: } cleanup normal {
// CIR-NEXT: cir.lifetime.end %[[TMP]]
// CIR: cir.call @after()
// LLVM-LABEL: define{{.*}} void @conditional_scalar(
// LLVM: call void @llvm.lifetime.start.p0(ptr %[[TMP:.*]])
// LLVM: br i1
// LLVM: call void @consume(ptr{{.*}} %[[TMP]])
// LLVM: call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// LLVM: call void @after()

// Precise lifetime boundaries must retain conditional starts and guarded ends.
// PRECISE-LABEL: cir.func{{.*}} @conditional_scalar(
// PRECISE: %[[TMP:.*]] = cir.alloca "ref.tmp0"
// PRECISE: %[[ACTIVE:.*]] = cir.alloca "cleanup.cond"
// PRECISE: cir.ternary
// PRECISE: cir.lifetime.start %[[TMP]]
// PRECISE: cir.call @consume(%[[TMP]])
// PRECISE: } cleanup normal {
// PRECISE: %[[FLAG:.*]] = cir.load{{.*}} %[[ACTIVE]]
// PRECISE: cir.if %[[FLAG]] {
// PRECISE-NEXT: cir.lifetime.end %[[TMP]]

// Both pending actions use the same stack. Guard the destructor and end
// separately, in reverse registration order.
extern "C" void conditional_guard(bool enabled) {
  (void)(enabled && (use_guard(Guard(3)), true));
  after();
}

// CIR-LABEL: cir.func{{.*}} @conditional_guard(
// CIR: %[[TMP:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!rec_Guard>
// CIR: %[[LIFE:.*]] = cir.alloca "cleanup.cond"{{.*}} : !cir.ptr<!cir.bool>
// CIR: %[[DTOR:.*]] = cir.alloca "cleanup.cond"{{.*}} : !cir.ptr<!cir.bool>
// CIR: cir.lifetime.start %[[TMP]]
// CIR: cir.call @_ZN5GuardC1Ei(%[[TMP]]
// CIR: cir.call @use_guard(%[[TMP]])
// CIR: } cleanup normal {
// CIR: %[[CHECK_DTOR:.*]] = cir.load{{.*}} %[[DTOR]]
// CIR: cir.if %[[CHECK_DTOR]] {
// CIR: cir.call @_ZN5GuardD1Ev(%[[TMP]])
// CIR: %[[CHECK_LIFE:.*]] = cir.load{{.*}} %[[LIFE]]
// CIR: cir.if %[[CHECK_LIFE]] {
// CIR: cir.lifetime.end %[[TMP]]
// CIR: cir.call @after()

// The deferred destructor and lifetime end also run in that order during
// unwinding, with their own activation flags.
// LLVM-EH-LABEL: define{{.*}} void @conditional_guard(
// LLVM-EH: %[[TMP:.*]] = alloca %struct.Guard
// LLVM-EH: %[[LIFE:.*]] = alloca i8
// LLVM-EH: %[[DTOR:.*]] = alloca i8
// LLVM-EH: call void @llvm.lifetime.start.p0(ptr %[[TMP]])
// LLVM-EH: store i8 1, ptr %[[LIFE]]
// LLVM-EH: invoke void @_ZN5GuardC1Ei(
// LLVM-EH: store i8 1, ptr %[[DTOR]]
// LLVM-EH: invoke void @use_guard(
// LLVM-EH: load i8, ptr %[[DTOR]]
// LLVM-EH: call void @_ZN5GuardD1Ev(ptr{{.*}} %[[TMP]])
// LLVM-EH: load i8, ptr %[[LIFE]]
// LLVM-EH: call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// LLVM-EH: landingpad
// LLVM-EH: load i8, ptr %[[DTOR]]
// LLVM-EH: call void @_ZN5GuardD1Ev(ptr{{.*}} %[[TMP]])
// LLVM-EH: load i8, ptr %[[LIFE]]
// LLVM-EH: call void @llvm.lifetime.end.p0(ptr %[[TMP]])
// LLVM-EH: resume
// LLVM-EH: call void @after()

// A conditionally initialized scalar reference extends the chosen temporary's
// lifetime. The end entries are promoted with their active flags intact.
extern "C" void conditional_extended(bool enabled) {
  const int &ref = enabled ? static_cast<const int &>(44)
                          : static_cast<const int &>(45);
  consume(ref);
  after();
}

// CIR-LABEL: cir.func{{.*}} @conditional_extended(
// CIR-DAG: %[[LEFT:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!s32i>
// CIR-DAG: %[[RIGHT:.*]] = cir.alloca "ref.tmp1"{{.*}} : !cir.ptr<!s32i>
// CIR: cir.lifetime.start %[[LEFT]]
// CIR-NOT: cir.lifetime.end
// CIR: cir.lifetime.start %[[RIGHT]]
// CIR-NOT: cir.lifetime.end
// CIR: cir.call @consume(
// CIR-NOT: cir.lifetime.end
// CIR: cir.call @after()
// CIR: cir.if %{{.*}} {
// CIR: cir.lifetime.end %[[RIGHT]]
// CIR: cir.if %{{.*}} {
// CIR: cir.lifetime.end %[[LEFT]]

struct Holder {
  int value;
};

// End the complete object's storage lifetime, rather than its member address.
extern "C" void subobject() {
  consume(Holder{46}.value);
  after();
}

// CIR-LABEL: cir.func{{.*}} @subobject()
// CIR: %[[TMP:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!rec_Holder>
// CIR: cir.lifetime.start %[[TMP]]
// CIR: cir.call @consume(
// CIR: cir.lifetime.end %[[TMP]] : !cir.ptr<!rec_Holder>
// CIR: cir.call @after()

using IntArray = int[2];

// A pending array lifetime end applies once to the entire alloca.
extern "C" void extended_array() {
  const IntArray &ref = IntArray{47, 48};
  consume(ref[0]);
  after();
}

// CIR-LABEL: cir.func{{.*}} @extended_array()
// CIR: %[[TMP:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!cir.array<!s32i x 2>>
// CIR: cir.lifetime.start %[[TMP]]
// CIR: cir.call @consume(
// CIR: cir.call @after()
// CIR: cir.lifetime.end %[[TMP]]

// Mixing unconditional and conditional temporaries keeps both destructor and
// lifetime-end order across the conditional scope.
extern "C" void mixed(bool enabled) {
  (use_guard(Guard(4)), (void)(enabled && (use_guard(Guard(5)), true)),
   use_guard(Guard(6)));
  after();
}

// CIR-LABEL: cir.func{{.*}} @mixed(
// CIR-DAG: %[[FIRST:.*]] = cir.alloca "ref.tmp0"{{.*}} : !cir.ptr<!rec_Guard>
// CIR-DAG: %[[SECOND:.*]] = cir.alloca "ref.tmp1"{{.*}} : !cir.ptr<!rec_Guard>
// CIR-DAG: %[[THIRD:.*]] = cir.alloca "ref.tmp2"{{.*}} : !cir.ptr<!rec_Guard>
// CIR: cir.call @use_guard(%[[THIRD]])
// CIR: cir.call @_ZN5GuardD1Ev(%[[THIRD]])
// CIR: cir.lifetime.end %[[THIRD]]
// CIR: cir.call @_ZN5GuardD1Ev(%[[SECOND]])
// CIR: cir.lifetime.end %[[SECOND]]
// CIR: cir.call @_ZN5GuardD1Ev(%[[FIRST]])
// CIR: cir.lifetime.end %[[FIRST]]
// CIR: cir.call @after()

// Hoisting nested branches must place both starts before the outer conditional.
// The starts execute on every loop iteration, and both ends precede after().
extern "C" void nested_loop(bool first, bool second) {
  while (first) {
    (void)(first && (second ? (consume(49), true) : (consume(50), false)));
    after();
  }
}

// CIR-LABEL: cir.func{{.*}} @nested_loop(
// CIR-NOT: cleanup.cond
// CIR: cir.while
// CIR: } do {
// CIR: cir.lifetime.start %[[LEFT:.*]] :
// CIR-NEXT: cir.lifetime.start %[[RIGHT:.*]] :
// CIR-NEXT: {{.*}}cir.ternary
// CIR: cir.call @consume(%[[LEFT]])
// CIR: cir.call @consume(%[[RIGHT]])
// CIR: } cleanup normal {
// CIR-NEXT: cir.lifetime.end %[[RIGHT]]
// CIR-NEXT: cir.lifetime.end %[[LEFT]]
// CIR: cir.call @after()

// Lifetime-extended entries from both arms are captured by the loop cleanup,
// keeping their branch flags rather than ending at the initializer boundary.
extern "C" void conditional_extended_loop(bool enabled) {
  while (const int &ref = enabled ? static_cast<const int &>(51)
                                 : static_cast<const int &>(0)) {
    consume(ref);
  }
}

// CIR-LABEL: cir.func{{.*}} @conditional_extended_loop(
// CIR-DAG: %[[LEFT:.*]] = cir.alloca "ref.tmp0"
// CIR-DAG: %[[RIGHT:.*]] = cir.alloca "ref.tmp1"
// CIR: cir.while {
// CIR: cir.lifetime.start %[[LEFT]]
// CIR-NOT: cir.lifetime.end
// CIR: cir.lifetime.start %[[RIGHT]]
// CIR-NOT: cir.lifetime.end
// CIR: cir.condition
// CIR: } do {
// CIR: cir.call @consume(
// CIR: } cleanup normal {
// CIR: cir.if %{{.*}} {
// CIR: cir.lifetime.end %[[RIGHT]]
// CIR: cir.if %{{.*}} {
// CIR: cir.lifetime.end %[[LEFT]]

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

// An enclosing catch requires an unwind edge and the marker must run before
// entering the handler.
void lifetime_in_try() {
  try {
    int x = 1;
    may_throw();
    use(x);
  } catch (...) {
  }
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z15lifetime_in_tryv()
// CIR-EH:         %[[X:.*]] = cir.alloca "x"
// CIR-EH:         cir.try {
// CIR-EH:           cir.lifetime.start %[[X]]
// CIR-EH:           cir.cleanup.scope {
// CIR-EH:             cir.call @_Z9may_throwv()
// CIR-EH:             cir.call @_Z3usei
// CIR-EH:           } cleanup all {
// CIR-EH:             cir.lifetime.end %[[X]]
// CIR-EH:         } catch all

// LLVM-EH-LABEL: define{{.*}} void @_Z15lifetime_in_tryv()
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[X:.*]])
// LLVM-EH:         invoke void @_Z9may_throwv()
// LLVM-EH:         invoke void @_Z3usei
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[X]])
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[X]])
// LLVM-EH:         call ptr @__cxa_begin_catch

// A real outer cleanup requires the inner lifetime marker on the EH path.
void lifetime_in_dtor_scope() {
  S s;
  int x = 1;
  use(x);
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z22lifetime_in_dtor_scopev()
// CIR-EH:         %[[S:.*]] = cir.alloca "s"
// CIR-EH:         %[[X:.*]] = cir.alloca "x"
// CIR-EH:         cir.lifetime.start %[[S]]
// CIR-EH:         cir.lifetime.start %[[X]]
// CIR-EH:         cir.call @_Z3usei
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.lifetime.end %[[X]]
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.call @_ZN1SD1Ev(%[[S]])
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.lifetime.end %[[S]]

// LLVM-EH-LABEL: define{{.*}} void @_Z22lifetime_in_dtor_scopev()
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[S:.*]])
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[X:.*]])
// LLVM-EH:         invoke void @_Z3usei
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[X]])
// LLVM-EH:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[S]])

// The first call needs no unwind edge. The inner destructor's existing EH
// path must still pass through both outer lifetime markers.
void lifetime_around_dtor_scope() {
  int x = 1;
  use(x);
  S s;
  use(x);
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z26lifetime_around_dtor_scopev()
// CIR-EH:         %[[X:.*]] = cir.alloca "x"
// CIR-EH:         %[[S:.*]] = cir.alloca "s"
// CIR-EH:         cir.lifetime.start %[[X]]
// CIR-EH:         cir.call @_Z3usei
// CIR-EH:         cir.lifetime.start %[[S]]
// CIR-EH:         cir.call @_Z3usei
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.call @_ZN1SD1Ev(%[[S]])
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.lifetime.end %[[S]]
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.lifetime.end %[[X]]

// LLVM-EH-LABEL: define{{.*}} void @_Z26lifetime_around_dtor_scopev()
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[X:.*]])
// LLVM-EH:         call void @_Z3usei
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[S:.*]])
// LLVM-EH:         invoke void @_Z3usei
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH:         call void @_ZN1SD1Ev(ptr {{.*}} %[[S]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[S]])
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[X]])

// The outer loop's destructor is still a loop cleanup region when the inner
// lifetime scope is flattened.
void lifetime_in_loop() {
  while (LoopCond c{}) {
    int x = 1;
    use(x);
  }
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z16lifetime_in_loopv()
// CIR-EH:         %[[C:.*]] = cir.alloca "c"
// CIR-EH:         cir.while {
// CIR-EH:         } do {
// CIR-EH:           %[[X:.*]] = cir.alloca "x"
// CIR-EH:           cir.lifetime.start %[[X]]
// CIR-EH:           cir.call @_Z3usei
// CIR-EH:           } cleanup all {
// CIR-EH:             cir.lifetime.end %[[X]]
// CIR-EH:         } cleanup all {
// CIR-EH:           cir.call @_ZN8LoopCondD1Ev(%[[C]])
// CIR-EH:           cir.lifetime.end %[[C]]

// LLVM-EH-LABEL: define{{.*}} void @_Z16lifetime_in_loopv()
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[X:.*]])
// LLVM-EH:         invoke void @_Z3usei
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[X]])
// LLVM-EH:         call void @_ZN8LoopCondD1Ev(ptr {{.*}} %[[C]])

// A marker-only loop cleanup also participates in an enclosing try's EH path.
void lifetime_loop_in_try() {
  try {
    while (int c = source())
      use(c);
  } catch (...) {
  }
}

// CIR-EH-LABEL: cir.func{{.*}} @_Z20lifetime_loop_in_tryv()
// CIR-EH:         cir.try {
// CIR-EH:           %[[C:.*]] = cir.alloca "c"
// CIR-EH:           cir.while {
// CIR-EH:             cir.call @_Z6sourcev()
// CIR-EH:           } do {
// CIR-EH:             cir.call @_Z3usei
// CIR-EH:           } cleanup all {
// CIR-EH:             cir.lifetime.end %[[C]]
// CIR-EH:         } catch all

// LLVM-EH-LABEL: define{{.*}} void @_Z20lifetime_loop_in_tryv()
// LLVM-EH:         call void @llvm.lifetime.start.p0(ptr %[[C:.*]])
// LLVM-EH:         invoke{{.*}} i32 @_Z6sourcev()
// LLVM-EH:         invoke void @_Z3usei
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])
// LLVM-EH:         landingpad { ptr, i32 }
// LLVM-EH:         call void @llvm.lifetime.end.p0(ptr %[[C]])
// LLVM-EH:         call ptr @__cxa_begin_catch

#endif // __EXCEPTIONS
