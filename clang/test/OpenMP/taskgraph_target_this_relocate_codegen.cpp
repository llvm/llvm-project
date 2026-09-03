// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -fopenmp-targets=x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

// A taskgraph in a member function.  A replay starts from the taskgraph
// construct's enclosing data environment (OpenMP 6.0 [14.3]), whose `this` is
// the object the function was called for this time, so a recorded target
// construct's relocation helper re-derives the base of every map entry reached
// through `this` from the `this` the taskgraph's record holds -- by value, so
// it is loaded from the record rather than taken as its address -- and moves
// the begin pointer with it.  That covers an explicit section of a member, the
// entry for the object a region maps implicitly to reach its members along
// with those members, and the same entries on a data directive.

struct S {
  int M;
  int Out[4];
  void section() {
#pragma omp taskgraph
    {
#pragma omp target map(tofrom : Out[0 : 4])
      Out[0] = 1;
    }
  }
  void implicit() {
#pragma omp taskgraph
    {
#pragma omp target
      Out[0] = M;
    }
  }
  void data() {
#pragma omp taskgraph
    {
#pragma omp target enter data map(to : Out[0 : 4])
    }
  }
};

void f(S &s) {
  s.section();
  s.implicit();
  s.data();
}

// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[SECTION:@.omp.taskgraph.target.relocate[.0-9]*]])
// Each entry is based on `this`, loaded from the record: base `this`, begin
// moved by the recorded distance.
// CHECK: define internal {{(noundef i32|void)}} [[SECTION]](ptr noundef %0, ptr noundef %1)
// CHECK: [[KA:%.+]] = load ptr, ptr %.addr,
// CHECK-NEXT: getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr [[KA]], i32 0, i32 2
// CHECK-NEXT: [[BASES:%.+]] = load ptr,
// CHECK-NEXT: getelementptr inbounds nuw %struct.__tgt_kernel_arguments, ptr [[KA]], i32 0, i32 3
// CHECK-NEXT: [[PTRS:%.+]] = load ptr,
// CHECK-NEXT: [[REC:%.+]] = load ptr, ptr %.addr1,
// CHECK-NEXT: [[THISF:%.+]] = getelementptr inbounds nuw %struct.anon{{[.0-9]*}}, ptr [[REC]], i32 0, i32 0
// CHECK-NEXT: [[THIS:%.+]] = load ptr, ptr [[THISF]],
// CHECK-NEXT: [[BSLOT:%.+]] = getelementptr ptr, ptr [[BASES]], i64 0
// CHECK-NEXT: [[PSLOT:%.+]] = getelementptr ptr, ptr [[PTRS]], i64 0
// CHECK-NEXT: [[OLDB:%.+]] = load ptr, ptr [[BSLOT]],
// CHECK-NEXT: [[OLDP:%.+]] = load ptr, ptr [[PSLOT]],
// CHECK-NEXT: [[P:%.+]] = ptrtoaddr ptr [[OLDP]] to i64
// CHECK-NEXT: [[B:%.+]] = ptrtoaddr ptr [[OLDB]] to i64
// CHECK-NEXT: [[D:%.+]] = sub i64 [[P]], [[B]]
// CHECK-NEXT: [[NEWP:%.+]] = getelementptr i8, ptr [[THIS]], i64 [[D]]
// CHECK-NEXT: store ptr [[THIS]], ptr [[BSLOT]],
// CHECK-NEXT: store ptr [[NEWP]], ptr [[PSLOT]],

// The entry for the object and its two members.
// CHECK: call i32 @__kmpc_taskgraph_target({{.*}}, ptr [[IMPLICIT:@.omp.taskgraph.target.relocate[.0-9]*]])
// CHECK: define internal {{(noundef i32|void)}} [[IMPLICIT]](ptr noundef %0, ptr noundef %1)
// CHECK-COUNT-3: getelementptr inbounds nuw %struct.anon{{[.0-9]*}}, ptr %{{[0-9]+}}, i32 0, i32 0
// CHECK-NOT: getelementptr inbounds nuw %struct.anon
// CHECK: ret {{i32 0|void}}

// The entry for the object and the member, on the data directive.
// CHECK: call void @__kmpc_taskgraph_target_enter_data({{.*}}, ptr [[DATA:@.omp.taskgraph.target.relocate[.0-9]*]])
// CHECK: define internal {{(noundef i32|void)}} [[DATA]](ptr noundef %0, ptr noundef %1, ptr noundef %2)
// CHECK-COUNT-2: getelementptr inbounds nuw %struct.anon{{[.0-9]*}}, ptr %{{[0-9]+}}, i32 0, i32 0
// CHECK-NOT: getelementptr inbounds nuw %struct.anon
// CHECK: ret {{i32 0|void}}

#endif
