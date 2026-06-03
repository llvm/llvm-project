// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

// Globals whose initializer takes the address of *another* global at a
// non-zero offset. The risk we are guarding against: when the referenced
// global's symType changes after its initializer is built (e.g. logical CIR
// record type swapped for an anon storage record), any GlobalViewAttr indices
// computed against the old type become stale and the resulting GEP lands on
// the wrong byte.

// Case 1: scalar -> scalar. Smoke test, no struct typing involved.
extern int b1;
int *a1 = &b1;
int b1 = 100;

// CIR-LABEL: cir.global external @a1
// CIR-SAME: #cir.global_view<@b1> : !cir.ptr<!s32i>
// LLVM: @a1 ={{.*}}ptr @b1
// OGCG: @a1 ={{.*}}ptr @b1

// Case 2: pointer into the interior of a later-emitted struct with an empty
// struct field. Tentative-definition ordering causes this one to "accidentally"
// work today because b2 gets emitted before a2, so a2 sees b2's final type.
struct E2 {};
struct B2 { struct E2 e; int x; int y; };
extern struct B2 b2;
int *a2 = &b2.y;
struct B2 b2 = { {}, 10, 20 };

// LLVM: @a2 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @b2, i64 4)
// OGCG: @a2 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @b2, i64 4)

// Case 3: pointer into a struct whose layout has an alignment-induced padding
// gap. Currently CIR computes the GEP offset against the anon-with-padding
// type of @b3 and lands on the padding bytes after `inner`, instead of inside
// `inner`. Reproduces the same type-swap mechanism as the self-reference 714
// case, applied across two globals.
struct E3 {};
struct In3 { int u, v; };
struct B3 {
  struct E3 e;
  int x;
  struct In3 inner;
  int *self;             // forces 4 bytes alignment padding before self
};
extern struct B3 b3;
int *a3 = &b3.inner.v;
struct B3 b3 = { .x = 7, .self = (int *)&b3 };

// LLVM: @a3 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @b3, i64 8)
// OGCG: @a3 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @b3, i64 8)

// Case 4: forward reference order — a is fully built before b's init runs, so
// any indices into b are necessarily captured against b's pre-init logical
// type. After b's const_record swaps b's symType to an anon storage record,
// the indices in a are stale.
struct E4 {};
struct In4 { int p, q; };
struct B4 {
  struct E4 e;
  int x;
  struct In4 inner;
  int *self;
};
extern struct B4 b4_fwd;
struct A4 { int *target; };
struct A4 a4 = { .target = &b4_fwd.inner.q };
struct B4 b4_fwd = { .x = 11, .self = (int *)&b4_fwd };

// LLVM: @a4 ={{.*}}getelementptr {{(inbounds nuw )?}}(i8, ptr @b4_fwd, i64 8)
// OGCG: @a4 ={{.*}}getelementptr {{(inbounds )?}}(i8, ptr @b4_fwd, i64 8)
