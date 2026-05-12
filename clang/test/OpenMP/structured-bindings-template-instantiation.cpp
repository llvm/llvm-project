// RUN: %clang_cc1 -verify -std=c++20 -fopenmp -triple x86_64-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s

// RUN: %clang_cc1 -verify -std=c++20 -fopenmp -triple x86_64-pc-linux-gnu -ast-dump %s | FileCheck %s --check-prefix=AST

// Test that template instantiation with structured binding captures in OpenMP
// works correctly when multiple bindings from the same decomposition are deduped
// into a single capture.

// expected-no-diagnostics

struct Point {
  int x, y;
};

struct Point3D {
  int x, y, z;
};


// CHECK-LABEL: define {{.*}}@_Z28test_template_single_binding5Point            
// CHECK: call void {{.*}}@__kmpc_fork_call(ptr {{.*}}, i32 1, ptr {{.*}}, ptr
template<typename T>
void test_template_single_binding(T p) {
  auto [a, b] = p;
#pragma omp parallel
  {
    use(a);  // Only one binding captured                                       
  }
}

// Template function capturing two bindings from struct decomposition.
template<typename T>
void test_template_two_bindings(T p) {
  auto [a, b] = p;
#pragma omp parallel reduction(+:result)
  {
    result = a + b;
  }
}

// Template function capturing three bindings.
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

// Template with multiple uses of same binding.
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

// Template with nested OpenMP constructs.
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

// Template with multiple OpenMP regions capturing same bindings.
template<typename T>
void test_template_multiple_regions(T p) {
  auto [a, b] = p;
  int result1 = 0, result2 = 0;
#pragma omp parallel reduction(+:result1)
  {
    result1 = a;
  }
}

void instantiate_tests() {
  Point p2{1, 2};
  Point3D p3{1, 2, 3};

  test_template_two_bindings(p2);
  test_template_two_bindings(Point{5, 6});

  test_template_three_bindings(p3);
  test_template_three_bindings(Point3D{7, 8, 9});

  test_template_reuse_bindings(p2);
  test_template_nested(p2);
  test_template_multiple_regions(p2);
}

typedef unsigned int size_t;

// Test with array bindings.
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

void test_array_instantiation() {
  int arr2[2] = {1, 2};
  test_template_array(arr2);
}
