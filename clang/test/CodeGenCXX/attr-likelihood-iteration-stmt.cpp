// RUN: %clang_cc1 -O1 -disable-llvm-passes -emit-llvm %s -o - -triple=x86_64-linux-gnu -verify
// RUN: %clang_cc1 -O1 -disable-llvm-passes -emit-llvm %s -o - -triple=x86_64-linux-gnu | FileCheck %s

void wl(int e){
  // CHECK-LABEL: define{{.*}}wl
  // CHECK: br {{.*}} !prof !6
  while(e) [[likely]] ++e;
}

void wu(int e){
  // CHECK-LABEL: define{{.*}}wu
  // CHECK: br {{.*}} !prof !10
  while(e) [[unlikely]] ++e;
}

void w_branch_elided(unsigned e){
  // CHECK-LABEL: define{{.*}}w_branch_elided
  // CHECK-NOT: br {{.*}} !prof
  // expected-warning@+2 {{attribute 'likely' has no effect when annotating an infinite loop}}
  // expected-note@+1 {{annotating the infinite loop here}}
  while(1) [[likely]] ++e;
}

void fl(unsigned e)
{
  // CHECK-LABEL: define{{.*}}fl
  // CHECK: br {{.*}} !prof !6
  for(int i = 0; i != e; ++e) [[likely]];
}

void fu(int e)
{
  // CHECK-LABEL: define{{.*}}fu
  // CHECK: br {{.*}} !prof !10
  for(int i = 0; i != e; ++e) [[unlikely]];
}

void f_branch_elided()
{
  // CHECK-LABEL: define{{.*}}f_branch_elided
  // CHECK-NOT: br {{.*}} !prof
  for(;;) [[likely]];
}

void frl(int (&&e) [4])
{
  // CHECK-LABEL: define{{.*}}frl
  // CHECK: br {{.*}} !prof !6
  for(int i : e) [[likely]];
}

void fru(int (&&e) [4])
{
  // CHECK-LABEL: define{{.*}}fru
  // CHECK: br {{.*}} !prof !10
  for(int i : e) [[unlikely]];
}

// CHECK: !6 = !{!"branch_weights", i32 2000, i32 1}
// CHECK: !10 = !{!"branch_weights", i32 1, i32 2000}
