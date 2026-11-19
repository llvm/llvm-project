// RUN: %clang_cc1 -verify -fopenmp -fsyntax-only %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fsyntax-only %s
// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// The data member is not captured until the template is instantiated.
struct B
{
  int i;
  template<typename T>
void foo();
};

template<typename T>
void B::foo()
{
#pragma omp parallel
  {
  #pragma omp sections lastprivate (i)
    {
      i = 0;
    }
  }
}

template void B::foo<int>();

int main() {
  B b;
  b.foo<int>();
  return b.i;
}

// CHECK-LABEL: define {{.*}}void @_ZN1B3fooIiEEvv(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{.+}}, i32 1, ptr @[[OUTLINED:_ZN1B3fooIiEEvv.omp_outlined]], ptr %{{.+}})

// CHECK: define internal void @[[OUTLINED]](
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_for_static_fini(
// CHECK: .omp.lastprivate.then:
// CHECK: .omp.lastprivate.done:
// CHECK: call void @__kmpc_barrier(
