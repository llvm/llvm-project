// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// A recorded 'omp target' keeps the map arrays it was launched with, so the
// addresses in them are the ones the mapped variables had on the encounter
// that recorded the graph.  A replay from a different frame finds them stale.
// __kmpc_taskgraph_target therefore takes a relocation callback, which the
// runtime calls with the recorded __tgt_kernel_arguments blob and the
// taskgraph record reconstructed for this encounter, and which rewrites the
// base-pointer and pointer arrays in place.
//
// Each slot holds two addresses, and the distance between them is a property
// of the map's shape rather than of the frame, so the callback re-derives only
// the base pointer and moves the begin pointer by the distance it finds in the
// recorded pair.  That is what lets a dynamic bound or a struct member through:
// the arithmetic that produced the begin pointer never has to be repeated.
// The callback is emitted only where every slot's base is reachable from a
// captured variable.

int G[64];

// Whole-variable maps: the base is the capture's live address and the recorded
// distance is zero, so base and begin come out the same.  The third slot is the
// implicit dyn_ptr argument, which names no variable.
void whole(void) {
  int A[64];
  int s = 0;
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : A) map(tofrom : s)
    { A[0] += s; }
  }
}

// A section: the base is still the variable, and the begin pointer is the
// variable plus whatever the recording put between them.
void constant_section(void) {
  int A[64];
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : A[4 : 10])
    { A[4] = 1; }
  }
}

// A variable with static storage duration is at the same address on every
// replay, so it is skipped rather than rewritten.  Nothing else here needs
// relocating either, and the callback is still emitted -- as a non-null no-op,
// because a null callback means "cannot relocate", not "nothing to relocate".
void only_global(void) {
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : G[0 : 64])
    { G[0] = 1; }
  }
}

// A pointer base: the mapped region begins at the *value* of p, so the base is
// reloaded from the capture rather than being its address.  The attach entry
// beside it is the other way round -- its base is the pointer object itself --
// and its pointee address is the one the map entry arrived at, since the two
// were emitted from a single address.
void pointer_base(int *p, int n) {
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : p[0 : n])
    { p[0] = 1; }
  }
}

// A dynamic lower bound is carried by the recorded distance, so nothing here
// depends on i being a constant.  What it does depend on is i not changing
// between replays, which is what `saved` is for: a moved bound is a change of
// shape, and a graph's extents and offsets are fixed when it is recorded.  The
// body's own firstprivate copy of i is another matter: it is a value, and it
// takes i's value at the replay.
void dynamic_bound(int i) {
  int A[64];
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : A[i : 10])
    { A[i] = 1; }
  }
}

// A struct member. Two slots share one base: the combined entry covering the
// mapped part of the struct, and the member-of entry inside it.
struct S {
  int A[64];
  int B;
  int *P;
};

void struct_member(void) {
  struct S s;
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : s.A[0 : 64])
    { s.A[0] = 1; }
  }
}

// A pointer reached through another pointer.  Re-deriving this base would mean
// loading `ps`, which is a value the replay has no licence to re-read: the
// graph froze it when it recorded the map.  Declined.
void double_indirection(struct S *ps, int n) {
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : ps->P[0 : n])
    { ps->P[0] = 1; }
  }
}

// A saved firstprivate aggregate's slot is repointed at a snapshot taken when
// the graph was recorded; rewriting it back to the live variable would undo
// what makes it saved.  Skipped, and since it is the only slot the callback is
// a no-op -- but a non-null one, so the maps beside such an aggregate still
// relocate.
void saved_aggregate(void) {
  struct S s;
#pragma omp taskgraph
  {
#pragma omp target firstprivate(saved : s)
    { s.B = 1; }
  }
}

// A firstprivate aggregate declared inside the region has no value in the
// environment a replay starts from -- the taskgraph construct's -- so its copy
// comes from the saved data environment without being asked for: the entry
// carries the saved bit exactly as saved_aggregate's does, and its slot is
// skipped the same way.
void inner_aggregate(int n) {
#pragma omp taskgraph
  {
    struct S s;
    s.B = n;
#pragma omp target firstprivate(s)
    { s.B += 1; }
  }
}

// A scalar the region captures by copy is a literal: its slot holds the value,
// not an address.  n is in the taskgraph's enclosing data environment, so its
// firstprivate takes n's value at the replay; the helper re-reads it through
// the taskgraph record and writes it into both slots at n's own width.  A
// global is re-read from the global.
int GS;
void literal_scalar(int n) {
  int A[64];
#pragma omp taskgraph
  {
#pragma omp target map(tofrom : A) firstprivate(GS)
    { A[0] = n + GS; }
  }
}

// None of these is re-read: a `saved` scalar keeps its recorded value, which
// its entry's SAVED bit says; a scalar declared inside the region has no value
// in the environment a replay starts from; and a clause argument's captured
// expression is evaluated when the graph is recorded.
void literal_kept(int n, int m) {
#pragma omp taskgraph
  {
    int k = n;
#pragma omp target teams num_teams(m) firstprivate(saved : n)
    { k += n; }
  }
}

// A 'target update' records map arrays of its own and gets a callback of its
// own, which takes the two arrays of addresses as its first two arguments
// rather than a kernel-args blob.  A strided section is the one shape where the
// recorded begin pointer is not an address in the mapped data at all but a
// pointer to the stride descriptor, which the runtime owns and copied when the
// graph was recorded -- so only the base is rewritten, and the descriptor's
// offsets, being measured from it, still name the same chunks.
void strided(void) {
  int A[8][8];
#pragma omp taskgraph
  {
#pragma omp target update to(A[1 : 2][3 : 4])
  }
}

// Each helper is emitted just after the region it belongs to, so these run in
// source order.  CHECK-LABEL cannot be used because it forbids the variables
// that tie a call site to the helper it names.

// Map types are emitted up front, also in source order.  The firstprivate
// entry of saved_aggregate and that of inner_aggregate are both
// PRIVATE | TO | TARGET_PARAM | SAVED.
// CHECK: @.offload_maptypes.{{[0-9]+}} = private unnamed_addr constant [2 x i64] [i64 65697, i64 288]
// CHECK: @.offload_maptypes.{{[0-9]+}} = private unnamed_addr constant [2 x i64] [i64 65697, i64 288]
// literal_scalar: A, then n (LITERAL | TARGET_PARAM | IMPLICIT), GS (LITERAL |
// TARGET_PARAM) and dyn_ptr.
// CHECK: @.offload_maptypes.{{[0-9]+}} = private unnamed_addr constant [4 x i64] [i64 35, i64 800, i64 288, i64 288]
// literal_kept: the inner k and the saved n are literals carrying SAVED; the
// num_teams expression's capture is a plain literal.
// CHECK: @.offload_maptypes.{{[0-9]+}} = private unnamed_addr constant [4 x i64] [i64 66336, i64 65824, i64 800, i64 288]

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[WHOLE:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// Field 2 of __tgt_kernel_arguments is ArgBasePtrs and field 3 is ArgPtrs; the
// sizes and map types beside them describe the shape of the maps, which
// relocation never changes.
// CHECK: define internal void [[WHOLE]](ptr noundef %0, ptr noundef %1)
// CHECK: [[KA:%.+]] = load ptr, ptr %.addr,
// CHECK: getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr [[KA]], i32 0, i32 2
// CHECK-NEXT: [[BASES:%.+]] = load ptr,
// CHECK-NEXT: getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr [[KA]], i32 0, i32 3
// CHECK-NEXT: [[PTRS:%.+]] = load ptr,
// CHECK-NEXT: [[REC:%.+]] = load ptr, ptr %.addr1,
// A: the base is the capture's live address, and the begin pointer is that
// address plus the distance the recorded pair still holds.
// CHECK-NEXT: getelementptr inbounds nuw %struct.anon, ptr [[REC]], i32 0, i32 0
// CHECK-NEXT: [[A:%.+]] = load ptr,
// CHECK-NEXT: [[AB:%.+]] = getelementptr ptr, ptr [[BASES]], i64 0
// CHECK-NEXT: [[AP:%.+]] = getelementptr ptr, ptr [[PTRS]], i64 0
// CHECK-NEXT: [[AOB:%.+]] = load ptr, ptr [[AB]],
// CHECK-NEXT: [[AOP:%.+]] = load ptr, ptr [[AP]],
// CHECK-NEXT: [[AE:%.+]] = ptrtoaddr ptr [[AOP]] to i64
// CHECK-NEXT: [[ABI:%.+]] = ptrtoaddr ptr [[AOB]] to i64
// CHECK-NEXT: [[AD:%.+]] = sub i64 [[AE]], [[ABI]]
// CHECK-NEXT: [[ANEW:%.+]] = getelementptr i8, ptr [[A]], i64 [[AD]]
// CHECK-NEXT: store ptr [[A]], ptr [[AB]],
// CHECK-NEXT: store ptr [[ANEW]], ptr [[AP]],
// s: likewise, in the next slot.
// CHECK-NEXT: getelementptr inbounds nuw %struct.anon, ptr [[REC]], i32 0, i32 1
// CHECK-NEXT: [[S:%.+]] = load ptr,
// CHECK-NEXT: [[SB:%.+]] = getelementptr ptr, ptr [[BASES]], i64 1
// CHECK-NEXT: [[SP:%.+]] = getelementptr ptr, ptr [[PTRS]], i64 1
// CHECK: store ptr [[S]], ptr [[SB]],
// CHECK-NEXT: store ptr {{%.+}}, ptr [[SP]],
// The dyn_ptr slot is a literal, so it is left alone.
// CHECK-NEXT: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[SECTION:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[SECTION]](ptr noundef %0, ptr noundef %1)
// CHECK: [[CAP:%.+]] = load ptr, ptr {{%.+}}, align 8, !nonnull
// CHECK-NEXT: [[CB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// CHECK-NEXT: [[CP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// CHECK: [[CNEW:%.+]] = getelementptr i8, ptr [[CAP]], i64 {{%.+}}
// CHECK-NEXT: store ptr [[CAP]], ptr [[CB]],
// CHECK-NEXT: store ptr [[CNEW]], ptr [[CP]],
// CHECK-NEXT: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[GLOBAL:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[GLOBAL]](ptr noundef %0, ptr noundef %1)
// No slot is indexed, so nothing is rewritten.
// CHECK-NOT: getelementptr ptr,
// CHECK: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[POINTER:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[POINTER]](ptr noundef %0, ptr noundef %1)
// The capture is the pointer object; the base is what it currently holds.
// CHECK: [[PVAR:%.+]] = load ptr, ptr {{%.+}}, align 8, !nonnull
// CHECK-NEXT: [[PVAL:%.+]] = load ptr, ptr [[PVAR]],
// CHECK-NEXT: [[PB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// CHECK-NEXT: [[PP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// CHECK: [[PNEW:%.+]] = getelementptr i8, ptr [[PVAL]], i64 {{%.+}}
// CHECK-NEXT: store ptr [[PVAL]], ptr [[PB]],
// CHECK-NEXT: store ptr [[PNEW]], ptr [[PP]],
// The attach entry: base is the pointer object itself, pointee is the address
// the map entry above arrived at, with no distance of its own.
// CHECK: [[AVAR:%.+]] = load ptr, ptr {{%.+}}, align 8, !nonnull
// CHECK-NEXT: [[TB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK-NEXT: [[TP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK-NEXT: store ptr [[AVAR]], ptr [[TB]],
// CHECK-NEXT: store ptr [[PNEW]], ptr [[TP]],
// CHECK-NEXT: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[DYNAMIC:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[DYNAMIC]](ptr noundef %0, ptr noundef %1)
// CHECK: [[DVAR:%.+]] = load ptr, ptr {{%.+}}, align 8, !nonnull
// CHECK-NEXT: [[DB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// CHECK-NEXT: [[DP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// The lower bound never appears; it is inside the recorded distance.
// CHECK: [[DNEW:%.+]] = getelementptr i8, ptr [[DVAR]], i64 {{%.+}}
// CHECK-NEXT: store ptr [[DVAR]], ptr [[DB]],
// CHECK-NEXT: store ptr [[DNEW]], ptr [[DP]],
// The body's own use of i is a by-copy literal in slot 1, and is re-read.
// CHECK-NEXT: getelementptr inbounds nuw %struct.anon{{[.0-9]*}}, ptr {{%.+}}, i32 0, i32 1
// CHECK-NEXT: [[IREF:%.+]] = load ptr,
// CHECK-NEXT: [[IVAL:%.+]] = load i32, ptr [[IREF]],
// CHECK-NEXT: [[IB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK-NEXT: [[IP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK-NEXT: store i32 [[IVAL]], ptr [[IB]],
// CHECK-NEXT: store i32 [[IVAL]], ptr [[IP]],
// CHECK-NEXT: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[MEMBER:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[MEMBER]](ptr noundef %0, ptr noundef %1)
// The combined entry for the struct, then the member-of entry, both based on
// the struct's capture.
// CHECK: [[MVAR:%.+]] = load ptr, ptr {{%.+}}, align 8, !nonnull
// CHECK-NEXT: [[MB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// CHECK-NEXT: [[MP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 0
// CHECK: store ptr [[MVAR]], ptr [[MB]],
// CHECK-NEXT: store ptr {{%.+}}, ptr [[MP]],
// CHECK: [[MVAR2:%.+]] = load ptr, ptr {{%.+}}, align 8, !nonnull
// CHECK-NEXT: [[MB2:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK-NEXT: [[MP2:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK: store ptr [[MVAR2]], ptr [[MB2]],
// CHECK-NEXT: store ptr {{%.+}}, ptr [[MP2]],
// CHECK-NEXT: ret void

// Declined, so the runtime keeps its refuse-to-relocate behaviour.
// CHECK: define {{.*}}@double_indirection
// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr null)

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[SAVED:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[SAVED]](ptr noundef %0, ptr noundef %1)
// CHECK-NOT: getelementptr ptr,
// CHECK: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[INNER:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[INNER]](ptr noundef %0, ptr noundef %1)
// CHECK-NOT: getelementptr ptr,
// CHECK: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[LITERAL:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[LITERAL]](ptr noundef %0, ptr noundef %1)
// Slot 0 is A's address, relocated as in whole().
// CHECK: getelementptr i8, ptr
// CHECK-NEXT: store ptr {{%.+}}, ptr {{%.+}},
// CHECK-NEXT: store ptr {{%.+}}, ptr {{%.+}},
// n, through the taskgraph record, at its own width in both slots.
// CHECK-NEXT: getelementptr inbounds nuw %struct.anon{{[.0-9]*}}, ptr {{%.+}}, i32 0, i32 1
// CHECK-NEXT: [[NREF:%.+]] = load ptr,
// CHECK-NEXT: [[NVAL:%.+]] = load i32, ptr [[NREF]],
// CHECK-NEXT: [[NB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK-NEXT: [[NP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 1
// CHECK-NEXT: store i32 [[NVAL]], ptr [[NB]],
// CHECK-NEXT: store i32 [[NVAL]], ptr [[NP]],
// GS, from the global.
// CHECK-NEXT: [[GVAL:%.+]] = load i32, ptr @GS,
// CHECK-NEXT: [[GB:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 2
// CHECK-NEXT: [[GP:%.+]] = getelementptr ptr, ptr {{%.+}}, i64 2
// CHECK-NEXT: store i32 [[GVAL]], ptr [[GB]],
// CHECK-NEXT: store i32 [[GVAL]], ptr [[GP]],
// CHECK-NEXT: ret void

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[KEPT:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[KEPT]](ptr noundef %0, ptr noundef %1)
// CHECK-NOT: getelementptr ptr,
// CHECK: ret void

// CHECK: call void @__kmpc_taskgraph_target_update({{.*}}, ptr [[STRIDED:@.omp.taskgraph.target.relocate[0-9.]*]])
//
// CHECK: define internal void [[STRIDED]](ptr noundef %0, ptr noundef %1, ptr noundef %2)
// A data node passes the recorded base-pointer array first and the pointer
// array second.
// CHECK: [[SBASES:%.+]] = load ptr, ptr %.addr,
// CHECK-NEXT: [[SPTRS:%.+]] = load ptr, ptr %.addr1,
// CHECK: [[SVAR:%.+]] = load ptr, ptr {{%.+}}, align 8, !nonnull
// CHECK-NEXT: [[SB:%.+]] = getelementptr ptr, ptr [[SBASES]], i64 0
// CHECK-NEXT: [[SP:%.+]] = getelementptr ptr, ptr [[SPTRS]], i64 0
// No recorded distance is read, and the slot holding the descriptor pointer is
// left as the runtime wrote it.
// CHECK-NEXT: store ptr [[SVAR]], ptr [[SB]],
// CHECK-NEXT: ret void

#endif
