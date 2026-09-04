// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -x c++ -std=c++20 \
// RUN: -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -verify -std=c++20 -fopenmp -triple x86_64-pc-linux-gnu \
// RUN: -ast-print %s | FileCheck %s --check-prefix=AST

// expected-no-diagnostics

// Test template instantiation with structured bindings in OpenMP regions.
// This verifies that skipping duplicate captures (when both bindings from
// the same decomposition are used) doesn't break template instantiation.

void use(int);

struct Point {
  int x, y;
};

struct Point3D {
  int x, y, z;
};

// CHECK-LABEL: define {{.*}} @_Z28test_template_single_bindingI5PointEvT_(
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr @{{[0-9]+}}, i32 1, ptr @{{.*}}.omp_outlined, ptr
//
// AST: void use(int);
// AST: struct Point {
// AST:     int x;
// AST:     int y;
// AST: };
// AST: struct Point3D {
// AST:     int x;
// AST:     int y;
// AST:     int z;
// AST: };
// AST: template <typename T> void test_template_single_binding(T p) {
// AST:     auto = p;
// AST:     #pragma omp parallel
// AST:         {
// AST:             use(a);
// AST:         }
// AST: }
// AST: template<> void test_template_single_binding<Point>(Point p) {
// AST:     auto = p;
// AST:     #pragma omp parallel
// AST:         {
// AST:             use(a);
// AST:         }
// AST: }
template<typename T>
void test_template_single_binding(T p) {
  auto [a, b] = p;
#pragma omp parallel
  {
    use(a);
  }
}

// CHECK-LABEL: define {{.*}}@_Z26test_template_two_bindingsI5PointEvT_
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr {{.*}}, ptr
//
// AST: template <typename T> void test_template_two_bindings(T p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = a + b;
// AST:         }
// AST: }
// AST: template<> void test_template_two_bindings<Point>(Point p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = a + b;
// AST:         }
// AST: }
template<typename T>
void test_template_two_bindings(T p) {
  auto [a, b] = p;
  int result = 0;
#pragma omp parallel reduction(+:result)
  {
    result = a + b;
  }
}

// CHECK-LABEL: define {{.*}}@_Z28test_template_three_bindingsI7Point3DEiT_
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr {{.*}}, ptr
//
// AST: template <typename T> int test_template_three_bindings(T p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = x + y + z;
// AST:         }
// AST:     return result;
// AST: }
// AST: template<> int test_template_three_bindings<Point3D>(Point3D p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = x + y + z;
// AST:         }
// AST:     return result;
// AST: }
template<typename T>
int test_template_three_bindings(T p) {
  auto [x, y, z] = p;

  int result = 0;
#pragma omp parallel reduction(+:result)
  {
    result = x + y + z;
  }
  return result;
}

// CHECK-LABEL: define {{.*}}@_Z28test_template_reuse_bindingsI5PointEiT_
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr {{.*}}, ptr
//
// AST: template <typename T> int test_template_reuse_bindings(T p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = a + b + a * 2 + b * 3;
// AST:         }
// AST:     return result;
// AST: }
// AST: template<> int test_template_reuse_bindings<Point>(Point p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = a + b + a * 2 + b * 3;
// AST:         }
// AST:     return result;
// AST: }
template<typename T>
int test_template_reuse_bindings(T p) {
  auto [a, b] = p;
  int result = 0;
#pragma omp parallel reduction(+:result)
  {
    result = a + b + a * 2 + b * 3;
  }
  return result;
}

// CHECK-LABEL: define {{.*}}@_Z20test_template_nestedI5PointEiT_
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr {{.*}}, ptr
//
// AST: template <typename T> int test_template_nested(T p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel
// AST:         {
// AST:             #pragma omp critical
// AST:                 {
// AST:                     result += a + b;
// AST:                 }
// AST:         }
// AST:     return result;
// AST: }
// AST: template<> int test_template_nested<Point>(Point p) {
// AST:     auto = p;
// AST:     int result = 0;
// AST:     #pragma omp parallel
// AST:         {
// AST:             #pragma omp critical
// AST:                 {
// AST:                     result += a + b;
// AST:                 }
// AST:         }
// AST:     return result;
// AST: }
template<typename T>
int test_template_nested(T p) {
  auto [a, b] = p;
  int result = 0;
#pragma omp parallel
  {
#pragma omp critical
    {
      result += a + b;
    }
  }
  return result;
}

// CHECK-LABEL: define {{.*}}@_Z30test_template_multiple_regionsI5PointEvT_
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr {{.*}}, ptr
//
// AST: template <typename T> void test_template_multiple_regions(T p) {
// AST:     auto = p;
// AST:     int result1 = 0, result2 = 0;
// AST:     #pragma omp parallel reduction(+: result1)
// AST:         {
// AST:             result1 = a;
// AST:         }
// AST: }
// AST: template<> void test_template_multiple_regions<Point>(Point p) {
// AST:     auto = p;
// AST:     int result1 = 0, result2 = 0;
// AST:     #pragma omp parallel reduction(+: result1)
// AST:         {
// AST:             result1 = a;
// AST:         }
// AST: }
template<typename T>
void test_template_multiple_regions(T p) {
  auto [a, b] = p;
  int result1 = 0, result2 = 0;
#pragma omp parallel reduction(+:result1)
  {
    result1 = a;
  }
}

typedef unsigned int size_t;
// CHECK-LABEL: define {{.*}}@_Z19test_template_arrayIiLj2EEiRAT0__T_
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 2, ptr {{.*}}, ptr
//
// AST: typedef unsigned int size_t;
// AST: template <typename T, size_t N> int test_template_array(T (&arr)[N]) {
// AST:     auto = arr;
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = a + b;
// AST:         }
// AST:     return result;
// AST: }
// AST: template<> int test_template_array<int, 2U>(int (&arr)[2]) {
// AST:     auto = {arr[*]};
// AST:     int result = 0;
// AST:     #pragma omp parallel reduction(+: result)
// AST:         {
// AST:             result = a + b;
// AST:         }
// AST:     return result;
// AST: }
template<typename T, size_t N>
int test_template_array(T (&arr)[N]) {
  auto [a, b] = arr;
  int result = 0;
#pragma omp parallel reduction(+:result)
  {
    result = a + b;
  }
  return result;
}

template<typename T>
struct Pair {
  T first;
  T second;
};

// CHECK-LABEL: define {{.*}}@_Z28test_dependent_decompositionI{{.*}}
// AST: template <typename T> void test_dependent_decomposition(Pair<T> p) {
// AST:    auto = p;
// AST:    #pragma omp parallel shared(a,b)
// AST:       {
// AST:            a = a + T(1);
// AST:            b = b + T(2);
// AST:        }
// AST: }
// AST: template<> void test_dependent_decomposition<int>(Pair<int> p) {
// AST:     auto = p;
// AST:     #pragma omp parallel shared(a,b)
// AST:         {
// AST:             a = a + int(1);
// AST:             b = b + int(2);
// AST:         }
// AST: }
// AST: template<> void test_dependent_decomposition<double>(Pair<double> p) {
// AST:         auto = p;
// AST:     #pragma omp parallel shared(a,b)
// AST:         {
// AST:             a = a + double(1);
// AST:             b = b + double(2);
// AST:         }
// AST: }

template<typename T>
void test_dependent_decomposition(Pair<T> p) {
  auto [a, b] = p;

#pragma omp parallel shared(a, b)
  {
    a = a + T(1);
    b = b + T(2);
  }
}

// CHECK-LABEL: define {{.*}}@_Z24test_dependent_reductionI{{.*}}
// AST: template <typename T> T test_dependent_reduction(Pair<T> p) {
// AST:     auto = p;
// AST:     T sum = T(0);
// AST:     #pragma omp parallel for reduction(+: sum)
// AST:         for (int i = 0; i < 10; ++i) {
// AST:             sum = sum + a + b;
// AST:         }
// AST:     return sum;
// AST: }
// AST: template<> int test_dependent_reduction<int>(Pair<int> p) {
// AST:     auto = p;
// AST:     int sum = int(0);
// AST:     #pragma omp parallel for reduction(+: sum)
// AST:         for (int i = 0; i < 10; ++i) {
// AST:             sum = sum + a + b;
// AST:         }
// AST:     return sum;
// AST: }
// AST: template<> double test_dependent_reduction<double>(Pair<double> p) {
// AST:     auto = p;
// AST:     double sum = double(0);
// AST:     #pragma omp parallel for reduction(+: sum)
// AST:         for (int i = 0; i < 10; ++i) {
// AST:             sum = sum + a + b;
// AST:         }
// AST:     return sum;
// AST: }
template<typename T>
T test_dependent_reduction(Pair<T> p) {
  auto [a, b] = p;
  T sum = T(0);

#pragma omp parallel for reduction(+:sum)
  for (int i = 0; i < 10; ++i) {
    sum = sum + a + b;
  }
  return sum;
}

// CHECK-LABEL: define {{.*}}@_Z27test_dependent_firstprivateI{{.*}}
// AST: template <typename T> void test_dependent_firstprivate(Pair<T> p) {
// AST:     auto = p;
// AST:     #pragma omp parallel firstprivate(a,b)
// AST:         {
// AST:             T local = a + b;
// AST:         }
// AST: }
// AST: template<> void test_dependent_firstprivate<int>(Pair<int> p) {
// AST:     auto = p;
// AST:     #pragma omp parallel firstprivate(a,b)
// AST:         {
// AST:             int local = a + b;
// AST:         }
// AST: }
// AST: template<> void test_dependent_firstprivate<double>(Pair<double> p) {
// AST:     auto = p;
// AST:     #pragma omp parallel firstprivate(a,b)
// AST:         {
// AST:             double local = a + b;
// AST:         }
// AST: }
template<typename T>
void test_dependent_firstprivate(Pair<T> p) {
  auto [a, b] = p;

#pragma omp parallel firstprivate(a, b)
  {
    T local = a + b;
  }
}

// CHECK-LABEL: define {{.*}}@_Z19test_dependent_taskI{{.*}}
// AST: template <typename T> void test_dependent_task(Pair<T> p) {
// AST:     auto = p;
// AST:     #pragma omp task shared(a)
// AST:         {
// AST:             a = a + T(10);
// AST:         }
// AST: }
// AST: template<> void test_dependent_task<int>(Pair<int> p) {
// AST:     auto = p;
// AST:     #pragma omp task shared(a)
// AST:         {
// AST:             a = a + int(10);
// AST:         }
// AST: }
// AST: template<> void test_dependent_task<double>(Pair<double> p) {
// AST:     auto = p;
// AST:     #pragma omp task shared(a)
// AST:         {
// AST:             a = a + double(10);
// AST:         }
// AST: }
template<typename T>
void test_dependent_task(Pair<T> p) {
  auto [a, b] = p;

#pragma omp task shared(a)
  {
    a = a + T(10);
  }
}

template<typename T>
struct Triple {
  T x, y, z;
};

// CHECK-LABEL: define {{.*}}@_Z30test_dependent_partial_captureIiEv6TripleIT_E
// AST: template <typename T> void test_dependent_partial_capture(Triple<T> p) {
// AST:     auto = p;
// AST:     #pragma omp parallel firstprivate(a)
// AST:         {
// AST:             T result = a + b + c;
// AST:         }
// AST: }
// AST: template<> void test_dependent_partial_capture<int>(Triple<int> p) {
// AST:     auto = p;
// AST:     #pragma omp parallel firstprivate(a)
// AST:         {
// AST:             int result = a + b + c;
// AST:         }
// AST: }
// AST: template<> void test_dependent_partial_capture<double>(Triple<double> p) {
// AST:     auto = p;
// AST:     #pragma omp parallel firstprivate(a)
// AST:         {
// AST:             double result = a + b + c;
// AST:         }
// AST: }
template<typename T>
void test_dependent_partial_capture(Triple<T> p) {
  auto [a, b, c] = p;
#pragma omp parallel firstprivate(a)
  {
    T result = a + b + c;
  }
}

// CHECK-LABEL: define {{.*}}@_Z32test_dependent_multiple_capturesI{{.*}}
// AST: template <typename T> void test_dependent_multiple_captures(Pair<T> p1, Pair<T> p2) {
// AST:     auto = p1;
// AST:     auto = p2;
// AST:     #pragma omp parallel shared(a,b,c,d)
// AST:         {
// AST:             T result = a + b + c + d;
// AST:         }
// AST: }
// AST: template<> void test_dependent_multiple_captures<int>(Pair<int> p1, Pair<int> p2) {
// AST:     auto = p1;
// AST:     auto = p2;
// AST:     #pragma omp parallel shared(a,b,c,d)
// AST:         {
// AST:             int result = a + b + c + d;
// AST:         }
// AST: }
// AST: template<> void test_dependent_multiple_captures<double>(Pair<double> p1, Pair<double> p2) {
// AST:     auto = p1;
// AST:     auto = p2;
// AST:     #pragma omp parallel shared(a,b,c,d)
// AST:         {
// AST:             double result = a + b + c + d;
// AST:         }
// AST: }
template<typename T>
void test_dependent_multiple_captures(Pair<T> p1, Pair<T> p2) {
  auto [a, b] = p1;
  auto [c, d] = p2;

#pragma omp parallel shared(a, b, c, d)
  {
    T result = a + b + c + d;
  }
}

void instantiate_tests() {
  Point p1{1, 2};
  Point3D p2{1, 2, 3};
  int arr[2] = {1, 2};
  test_template_single_binding(p1);
  test_template_two_bindings(p1);
  test_template_three_bindings(p2);
  test_template_reuse_bindings(p1);
  test_template_nested(p1);
  test_template_multiple_regions(p1);
  test_template_array(arr);

  Pair<int> pi{1, 2};
  test_dependent_decomposition(pi);
  test_dependent_reduction(pi);
  test_dependent_firstprivate(pi);
  test_dependent_task(pi);

  Pair<int> pi2{3, 4};
  test_dependent_multiple_captures(pi, pi2);

  Triple<int> ti{1, 2, 3};
  test_dependent_partial_capture(ti);

  Pair<double> pd{1.5, 2.5};
  test_dependent_decomposition(pd);
  test_dependent_reduction(pd);
  test_dependent_firstprivate(pd);
  test_dependent_task(pd);

  Pair<double> pd2{3.5, 4.5};
  test_dependent_multiple_captures(pd, pd2);

  Triple<double> td{1.5, 2.5, 3.5};
  test_dependent_partial_capture(td);
}
