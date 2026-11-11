// RUN: %clang_cc1 -verify -fopenmp -std=c++23 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp-simd -std=c++23 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s | FileCheck %s --check-prefix=SIMD-ONLY0
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
// expected-no-diagnostics

// C++ range-based for loops with class-type iterators under OpenMP loop
// directives. The precondition Sema builds for such a loop is the loop's own
// `__begin != __end` test when the iterator has no `operator<`, so CodeGen has
// to emit `__begin` along with `__range` and `__end` before the pre-inits
// (GH199449), and it has to do so for every loop whose bounds Sema hoists,
// which includes the loops that an `ordered(n)` clause counts but `collapse`
// does not.

typedef __PTRDIFF_TYPE__ ptrdiff_t;

template <int N>
struct int_range {
  struct iter {
    int v;
    int operator*() const { return v; }
    iter &operator++() {
      ++v;
      return *this;
    }
    iter operator++(int) {
      iter t = *this;
      ++v;
      return t;
    }
    bool operator!=(const iter &o) const { return v != o.v; }
    int operator-(const iter &o) const { return v - o.v; }
    iter operator+(ptrdiff_t n) const { return {v + (int)n}; }
    iter &operator+=(ptrdiff_t n) {
      v += (int)n;
      return *this;
    }
  };
  iter begin() const { return {0}; }
  iter end() const { return {N}; }
};

// The reproducer of GH199449: collapse(2) over two range-for loops in a
// function template, with a simd reduction in the body.
template <int M, int N, int K>
void matmul(const float *__restrict__ A, const float *__restrict__ B,
            float *__restrict__ C) {
#pragma omp parallel for collapse(2) schedule(static)
  for (auto ii : int_range<M>()) {
    for (auto jj : int_range<N>()) {
      float sum = 0.f;
#pragma omp simd reduction(+ : sum)
      for (int k = 0; k < K; ++k)
        sum += A[ii * K + k] * B[jj * K + k];
      C[ii * N + jj] = sum;
    }
  }
}

void test(const float *A, const float *B, float *C) {
  matmul<64, 512, 512>(A, B, C);
}
// CHECK-LABEL: define {{.*}}@_Z6matmulILi64ELi512ELi512EEv
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 {{[0-9]+}}, ptr @[[MATMUL:[^,]+]],
// CHECK: define internal void @[[MATMUL]](
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi64EE5beginEv(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi512EE5beginEv(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi64EE4itermiE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi512EE4itermiE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi64EE4iterneE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi512EE4iterneE
// CHECK-DAG: call void @__kmpc_for_static_init_{{4|8}}(
// CHECK-DAG: call {{.*}}@_ZN{{K?}}9int_rangeILi64EE4iterp{{[lL]}}E
// CHECK-DAG: call {{.*}}@_ZN{{K?}}9int_rangeILi512EE4iterp{{[lL]}}E
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi64EE4iterdeEv(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi512EE4iterdeEv(
// CHECK: call void @__kmpc_for_static_fini(

// The same nest outside of a template.
void matmul_plain(const float *__restrict__ A, const float *__restrict__ B,
                  float *__restrict__ C) {
#pragma omp parallel for collapse(2) schedule(static)
  for (auto ii : int_range<64>()) {
    for (auto jj : int_range<512>()) {
      float sum = 0.f;
#pragma omp simd reduction(+ : sum)
      for (int k = 0; k < 512; ++k)
        sum += A[ii * 512 + k] * B[jj * 512 + k];
      C[ii * 512 + jj] = sum;
    }
  }
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}matmul_plain
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 {{[0-9]+}}, ptr @[[PLAIN:[^,]+]],
// CHECK: define internal void @[[PLAIN]](
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi64EE4iterneE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi512EE4iterneE
// CHECK: call void @__kmpc_for_static_init_{{4|8}}(
// CHECK: call void @__kmpc_for_static_fini(

// The minimal shape: a single range-for loop, no collapse, no nested directive.
void single_range(float *C) {
#pragma omp for
  for (auto ii : int_range<4>())
    C[ii] = 0;
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}single_rangePf(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi4EE5beginEv(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi4EE4itermiE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi4EE4iterneE
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call {{.*}}@_ZN{{K?}}9int_rangeILi4EE4iterp{{[lL]}}E
// CHECK: call {{.*}}@_ZNK9int_rangeILi4EE4iterdeEv(
// CHECK: call void @__kmpc_for_static_fini(

// A loop that 'ordered(2)' counts but 'collapse' does not is emitted as an
// ordinary loop in the body, yet its trip count is hoisted for the doacross
// initialization, so its __range, __begin and __end are needed up front too.
void doacross_range_range(float *C) {
#pragma omp parallel for ordered(2)
  for (auto ii : int_range<4>())
    for (auto jj : int_range<8>())
      C[ii * 8 + jj] = 0;
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}doacross_range_rangePf(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 {{[0-9]+}}, ptr @[[RANGE_RANGE:[^,]+]],
// CHECK: define internal void @[[RANGE_RANGE]](
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi4EE4iterneE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi8EE5beginEv(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi8EE4itermiE
// CHECK-DAG: call void @__kmpc_doacross_init(ptr @{{[0-9]+}}, i32 %{{.+}}, i32 2, ptr %{{.+}})
// CHECK-DAG: call void @__kmpc_for_static_init_{{4|8}}(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi8EE4iterneE
// CHECK-DAG: call {{.*}}@_ZN9int_rangeILi8EE4iterppEv(
// CHECK: call void @__kmpc_doacross_fini(

void doacross_plain_range(float *C, int n) {
#pragma omp parallel for ordered(2)
  for (int ii = 0; ii < n; ++ii)
    for (auto jj : int_range<8>())
      C[ii * 8 + jj] = 0;
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}doacross_plain_rangePfi(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 {{[0-9]+}}, ptr @[[PLAIN_RANGE:[^,]+]],
// CHECK: define internal void @[[PLAIN_RANGE]](
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi8EE5beginEv(
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi8EE4itermiE
// CHECK-DAG: call void @__kmpc_doacross_init(ptr @{{[0-9]+}}, i32 %{{.+}}, i32 2, ptr %{{.+}})
// CHECK-DAG: call {{.*}}@_ZN9int_rangeILi8EE4iterppEv(
// CHECK: call void @__kmpc_doacross_fini(

// The same with an array range, where the hoisted __begin is a pointer.
void doacross_carray(float *C, int n, int (&arr)[8]) {
#pragma omp for ordered(2)
  for (int ii = 0; ii < n; ++ii)
    for (int jj : arr)
      C[ii * 8 + jj] = 0;
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}doacross_carrayPfiRA8_i(
// CHECK: call void @__kmpc_doacross_init(ptr @{{[0-9]+}}, i32 %{{.+}}, i32 2, ptr %{{.+}})
// CHECK: call void @__kmpc_for_static_init_4(
// CHECK: call void @__kmpc_doacross_fini(

// An init-statement is hoisted along with the range of the loop it belongs to.
void doacross_init_stmt(float *C, int n) {
#pragma omp parallel for ordered(2)
  for (int ii = 0; ii < n; ++ii)
    for (int base = 1; auto jj : int_range<8>())
      C[(ii + base) * 8 + jj] = 0;
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}doacross_init_stmtPfi(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 {{[0-9]+}}, ptr @[[INIT_STMT:[^,]+]],
// CHECK: define internal void @[[INIT_STMT]](
// CHECK-DAG: call void @__kmpc_doacross_init(ptr @{{[0-9]+}}, i32 %{{.+}}, i32 2, ptr %{{.+}})
// CHECK-DAG: call {{.*}}@_ZN9int_rangeILi8EE4iterppEv(
// CHECK: call void @__kmpc_doacross_fini(

// Two collapsed range-for loops plus one that only 'ordered(3)' counts.
void doacross_collapse2_ordered3(float *C) {
#pragma omp parallel for collapse(2) ordered(3)
  for (auto ii : int_range<4>())
    for (auto jj : int_range<8>())
      for (auto kk : int_range<16>())
        C[(ii * 8 + jj) * 16 + kk] = 0;
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}doacross_collapse2_ordered3Pf(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 {{[0-9]+}}, ptr @[[COLLAPSE2_ORDERED3:[^,]+]],
// CHECK: define internal void @[[COLLAPSE2_ORDERED3]](
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi4EE4iterneE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi8EE4iterneE
// CHECK-DAG: call {{.*}}@_ZNK9int_rangeILi16EE4itermiE
// CHECK-DAG: call void @__kmpc_doacross_init(ptr @{{[0-9]+}}, i32 %{{.+}}, i32 3, ptr %{{.+}})
// CHECK-DAG: call {{.*}}@_ZN9int_rangeILi16EE4iterppEv(
// CHECK: call void @__kmpc_doacross_fini(

// Control: with an operator< the precondition is built from the captured
// bounds instead of the loop's own condition.
struct lt_range {
  struct iter {
    int v;
    int operator*() const { return v; }
    iter &operator++() {
      ++v;
      return *this;
    }
    bool operator!=(const iter &o) const { return v != o.v; }
    bool operator<(const iter &o) const { return v < o.v; }
    int operator-(const iter &o) const { return v - o.v; }
    iter &operator+=(ptrdiff_t n) {
      v += (int)n;
      return *this;
    }
  };
  iter begin() const { return {0}; }
  iter end() const { return {8}; }
};

void lt_control(float *C) {
#pragma omp parallel for collapse(2)
  for (auto ii : lt_range())
    for (auto jj : lt_range())
      C[ii * 8 + jj] = 0;
}
// CHECK-LABEL: define {{.*}}@_Z{{[0-9]+}}lt_controlPf(
// CHECK: call void (ptr, i32, ptr, ...) @__kmpc_fork_call(ptr @{{[0-9]+}}, i32 {{[0-9]+}}, ptr @[[LT:[^,]+]],
// CHECK: define internal void @[[LT]](
// CHECK: call {{.*}}@_ZNK8lt_range4iterltE
// CHECK: call void @__kmpc_for_static_init_{{4|8}}(
// CHECK: call void @__kmpc_for_static_fini(
