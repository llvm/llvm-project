// RUN: %clang_cc1 -fopenmp -x c++ -triple x86_64-apple-darwin10  -stack-protector 2 -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple x86_64-apple-darwin10  -stack-protector 2 -emit-llvm -o - %s | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Check that function attributes are added to the OpenMP runtime functions.

template <class T>
struct S {
  T f;
  S(T a) : f(a) {}
  S() : f() {}
  operator T() { return T(); }
  ~S() {}
};

// CHECK: define internal void @.omp.copyprivate.copy_func(ptr noundef %0, ptr noundef %1) [[ATTR0:#[0-9]+]] {

void foo0();

int foo1() {
  char a;

#pragma omp parallel
  a = 2;
#pragma omp single copyprivate(a)
  foo0();

  return 0;
}

// CHECK: define internal void @.omp_task_privates_map.({{.*}}) [[ATTR3:#[0-9]+]] {
// CHECK: define internal noundef i32 @.omp_task_entry.({{.*}}) [[ATTR0]] {
// CHECK: define internal noundef i32 @.omp_task_destructor.({{.*}}) [[ATTR0]] {

int foo2() {
  S<double> s_arr[] = {1, 2};
  S<double> var(3);
#pragma omp task private(s_arr, var)
  s_arr[0] = var;
  return 0;
}

// CHECK: define internal void @.omp.reduction.reduction_func(ptr noundef %0, ptr noundef %1) [[ATTR0]] {

float foo3(int n, float *a, float *b) {
  int i;
  float result;

#pragma omp parallel for private(i) reduction(+:result)
  for (i=0; i < n; i++)
    result = result + (a[i] * b[i]);
  return result;
}

// CHECK: attributes [[ATTR0]] = {{{.*}} sspstrong {{.*}}}
// CHECK: attributes [[ATTR3]] = {{{.*}} sspstrong {{.*}}}
