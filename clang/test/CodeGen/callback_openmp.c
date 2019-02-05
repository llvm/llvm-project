// RUN: %clang_cc1 -triple i386-unknown-unknown -fopenmp -O1 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -fopenmp -O1 %s -emit-llvm -o - | opt -ipconstprop -S | FileCheck --check-prefix=IPCP %s

// CHECK: declare !callback ![[cid:[0-9]+]] void @__kmpc_fork_call
// CHECK: declare !callback ![[cid]] void @__kmpc_fork_teams
// CHECK: ![[cid]] = !{![[cidb:[0-9]+]]}
// CHECK: ![[cidb]] = !{i64 2, i64 -1, i64 -1, i1 true}

void work1(int, int);
void work2(int, int);
void work12(int, int);

void foo(int q) {
  int p = 2;

  #pragma omp parallel firstprivate(q, p)
  work1(p, q);
// IPCP: call void @work1(i32 2, i32 %{{[._a-zA-Z0-9]*}})

  #pragma omp parallel for firstprivate(p, q)
  for (int i = 0; i < q; i++)
    work2(i, p);
// IPCP: call void @work2(i32 %{{[._a-zA-Z0-9]*}}, i32 2)

  #pragma omp target teams firstprivate(p)
  work12(p, p);
// IPCP: call void @work12(i32 2, i32 2)
}
