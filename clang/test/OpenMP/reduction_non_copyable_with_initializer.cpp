// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -x c++ -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// expected-no-diagnostics

// Simple wrapper class that requires construction before assignment
struct Wrapper {
  int* ptr;
  Wrapper() : ptr(new int(0)) {}
  ~Wrapper() { delete ptr; }

  // Non-copyable
  Wrapper(const Wrapper&) = delete;
  Wrapper& operator=(const Wrapper&) = delete;

  void assign(int val) { *ptr = val; }
  int get() const { return *ptr; }
};

struct my_struct {
  int a;
  int b;
  Wrapper w; // Non-trivial member that needs construction

  my_struct() : a(1), b(1), w() {}

  // Non-copyable: deleted copy constructor and assignment
  my_struct(const my_struct&) = delete;
  my_struct& operator=(const my_struct&) = delete;
};

void my_init_default(my_struct& t) {
  t.a = 0;
  t.b = 0;
  t.w.assign(0); // This requires w to be constructed first
}

void my_add(my_struct& lhs, const my_struct& rhs) {
  lhs.a += rhs.a;
  lhs.b += rhs.b;
}

static my_struct x;
my_struct y;

// Custom reduction with user-defined initializer.
// The initializer uses operations that require the object to be properly
// constructed first (especially the non-trivial Wrapper member).
// Without emitting the default constructor first, this would crash.
#pragma omp declare reduction(my_reduction_add : my_struct : my_add(omp_out, omp_in)) \
    initializer(my_init_default(omp_priv))

void foo() {
  #pragma omp parallel reduction(my_reduction_add:y)
  my_add(y, x);
}

// Verify that the .omp_initializer function emits the default constructor
// BEFORE calling the user-defined initializer function. This is critical for
// types with non-trivial members (like Wrapper with its pointer management).
// Without the constructor call first, the user initializer would operate on
// uninitialized memory, causing crashes.

// CHECK-LABEL: define internal void @.omp_initializer.
// CHECK: call {{.*}} @_ZN9my_structC1Ev(ptr {{[^)]+}})
// CHECK-NEXT: call void @_Z15my_init_defaultR9my_struct(ptr {{[^)]+}})
// CHECK-NEXT: ret void
