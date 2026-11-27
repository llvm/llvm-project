// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefixes=LLVM,LLVMCIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefixes=LLVM,OGCG

// Large, mostly zero aggregate initializers are emitted as a memset of the
// whole object followed by stores of the non-zero parts. This also zeroes the
// padding.

struct Big {
  int a;      // followed by 4 bytes of padding
  long b[32];
  int c;      // followed by 4 bytes of padding
};

void assign_empty(Big *p) { *p = {}; }

// CIR-LABEL: cir.func {{.*}} @_Z12assign_emptyP3Big
// CIR:         %[[TMP:.*]] = cir.alloca "ref.tmp0"
// CIR:         %[[VOID:.*]] = cir.cast bitcast %[[TMP]] : !cir.ptr<!rec_Big> -> !cir.ptr<!void>
// CIR-NEXT:    %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CIR-NEXT:    %[[SIZE:.*]] = cir.const #cir.int<272> : !u64i
// CIR-NEXT:    cir.libc.memset %[[SIZE]] bytes at %[[VOID]] align(8) to %[[ZERO]]
// CIR-NOT:     cir.store
// CIR:         cir.call @_ZN3BigaSEOS_({{.*}}, %[[TMP]])

// OGCG avoids emitting the trivial copy/move assignment operator when it is not
// necessary, CIR have chosen to emit it and allow later optimizations to remove
// it when appropriate. See also the comment in
// CIRGenFunction::emitCXXMemberOrOperatorMemberCallExpr.
// LLVM-LABEL: define {{.*}} void @_Z12assign_emptyP3Big
// LLVM:         call void @llvm.memset.p0.i64(ptr align 8 %[[TMP:.*]], i8 0, i64 272, i1 false)
// LLVM-NOT:     store
// LLVMCIR:      call {{.*}} @_ZN3BigaSEOS_({{.*}}, ptr {{.*}} %[[TMP]])
// OGCG:         call void @llvm.memcpy.p0.p0.i64(ptr {{.*}}, ptr align 8 %[[TMP]], i64 272, i1 false)

void init_mostly_zero(int v) { Big b = {v}; }

// CIR-LABEL: cir.func {{.*}} @_Z16init_mostly_zeroi
// CIR:         %[[B:.*]] = cir.alloca "b"
// CIR:         %[[VOID:.*]] = cir.cast bitcast %[[B]] : !cir.ptr<!rec_Big> -> !cir.ptr<!void>
// CIR-NEXT:    %[[ZERO:.*]] = cir.const #cir.int<0> : !u8i
// CIR-NEXT:    %[[SIZE:.*]] = cir.const #cir.int<272> : !u64i
// CIR-NEXT:    cir.libc.memset %[[SIZE]] bytes at %[[VOID]] align(8) to %[[ZERO]]
// CIR-NEXT:    %[[A:.*]] = cir.get_member %[[B]][0] {name = "a"}
// CIR-NEXT:    %[[V:.*]] = cir.load {{.*}} : !cir.ptr<!s32i>, !s32i
// CIR-NEXT:    cir.store {{.*}} %[[V]], %[[A]]
// CIR-NOT:     cir.store
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} void @_Z16init_mostly_zeroi
// LLVM:         call void @llvm.memset.p0.i64(ptr align 8 %{{.*}}, i8 0, i64 272, i1 false)
// LLVM:         store i32
// LLVM-NOT:     store
// LLVM:         ret void

// 16 bytes or less: individual stores, no memset.
struct Small {
  long x, y;
};

void init_small(long v) { Small s = {v}; }

// CIR-LABEL: cir.func {{.*}} @_Z10init_smalll
// CIR-NOT:     cir.libc.memset
// CIR:         %[[Y:.*]] = cir.get_member %{{.*}}[1] {name = "y"}
// CIR-NEXT:    %[[ZERO:.*]] = cir.const #cir.int<0> : !s64i
// CIR-NEXT:    cir.store {{.*}} %[[ZERO]], %[[Y]]
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} void @_Z10init_smalll
// LLVM-NOT:     call void @llvm.memset
// LLVM:         store i64 0
// LLVM:         ret void

// More than a quarter of the bytes are non-zero: no memset.
struct Mixed {
  long a, b, c, d;
};

void init_mixed(long v) { Mixed m = {v, v, 0, 0}; }

// CIR-LABEL: cir.func {{.*}} @_Z10init_mixedl
// CIR-NOT:     cir.libc.memset
// CIR:         cir.return

// LLVM-LABEL: define {{.*}} void @_Z10init_mixedl
// LLVM-NOT:     call void @llvm.memset
// LLVM:         ret void
