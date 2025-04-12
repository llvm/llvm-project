// RUN: %clang_cc1 %s -Wno-strict-prototypes -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -Wno-strict-prototypes -triple amdgcn-amd-amdhsa -emit-llvm -o - | FileCheck --check-prefix=NONZEROALLOCAAS %s

struct abc {
 long a;
 long b;
 long c;
 long d;
 long e;
 long f;
 long g;
 long h;
 long i;
 long j;
};

struct abc foo1(void);
// CHECK-DAG: declare {{.*}} @foo1(ptr dead_on_unwind writable sret(%struct.abc)
// NONZEROALLOCAAS-DAG: declare {{.*}} @foo1(ptr addrspace(5) dead_on_unwind writable sret(%struct.abc)
struct abc foo2();
// CHECK-DAG: declare {{.*}} @foo2(ptr dead_on_unwind writable sret(%struct.abc)
// NONZEROALLOCAAS-DAG: declare {{.*}} @foo2(ptr addrspace(5) dead_on_unwind writable sret(%struct.abc)
struct abc foo3(void) { return (struct abc){0}; }
// CHECK-DAG: define {{.*}} @foo3(ptr dead_on_unwind noalias writable sret(%struct.abc)
// NONZEROALLOCAAS-DAG: define {{.*}} @foo3(ptr addrspace(5) dead_on_unwind noalias writable sret(%struct.abc)

void bar(void) {
  struct abc dummy1 = foo1();
  // CHECK-DAG: call {{.*}} @foo1(ptr dead_on_unwind writable sret(%struct.abc)
  // NONZEROALLOCAAS-DAG: call {{.*}} @foo1(ptr addrspace(5) dead_on_unwind writable sret(%struct.abc)
  struct abc dummy2 = foo2();
  // CHECK-DAG: call {{.*}} @foo2(ptr dead_on_unwind writable sret(%struct.abc)
  // NONZEROALLOCAAS-DAG: call {{.*}} @foo2(ptr addrspace(5) dead_on_unwind writable sret(%struct.abc)
}
