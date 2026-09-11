// RUN: %clang_cc1 -Wno-error=incompatible-pointer-types -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -Wno-error=incompatible-pointer-types -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -Wno-error=incompatible-pointer-types -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// A `static` local can't be a VLA, but it can have a variably modified type --
// a pointer to a VLA. The variable itself is an ordinary global; the VLA bound
// still has to be evaluated in the enclosing function, for its side effects.

int bound(void);
int buf[100];

// The two static locals are ordinary globals initialized to &buf. Both
// functions emit their globals into the same module prologue, so check them
// together and out of order.

// CIR-DAG: cir.global "private" internal dso_local @side_effecting_bound.p = #cir.global_view<@buf> : !cir.ptr<!s32i>
// CIR-DAG: cir.global "private" internal dso_local @vararrays_probe.q = #cir.global_view<@buf> : !cir.ptr<!s32i>

// LLVM-DAG: @side_effecting_bound.p = internal global ptr @buf
// LLVM-DAG: @vararrays_probe.q = internal global ptr @buf

// OGCG-DAG: @side_effecting_bound.p = internal global ptr @buf
// OGCG-DAG: @vararrays_probe.q = internal global ptr @buf

void side_effecting_bound(void) {
  static int (*p)[bound()] = &buf;
  (void)p;
}

// CIR-LABEL: cir.func{{.*}} @side_effecting_bound()
// CIR:         cir.call @bound() : () -> !s32i

// LLVM-LABEL: define {{.*}} void @side_effecting_bound()
// LLVM:         call i32 @bound()

// OGCG-LABEL: define {{.*}} void @side_effecting_bound()
// OGCG:         call i32 @bound()

// This is autoconf's AC_C_VARARRAYS probe. It mainly has to compile at all; a
// failure here silently flips HAVE_C_VARARRAYS in every configure-based project.

int vararrays_probe(int m, int c[m][m]) {
  static int (*q)[m] = &buf;
  return c && q != 0;
}

// CIR-LABEL: cir.func{{.*}} @vararrays_probe(
// CIR:         cir.get_global @vararrays_probe.q : !cir.ptr<!cir.ptr<!s32i>>

// LLVM-LABEL: define {{.*}} i32 @vararrays_probe(
// LLVM:         load ptr, ptr @vararrays_probe.q

// OGCG-LABEL: define {{.*}} i32 @vararrays_probe(
// OGCG:         load ptr, ptr @vararrays_probe.q
