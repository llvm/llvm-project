// RUN: %clang_cc1 -verify -fopenmp -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s -ast-print -o - -Wno-source-uses-openmp -Wno-openmp-clauses | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -std=c++14 -fexceptions -fcxx-exceptions %s -ast-print -o - -Wno-source-uses-openmp -Wno-openmp-clauses | FileCheck %s

// expected-no-diagnostics

// CHECK: int foo();
int foo();

// CHECK:      template <typename T> T foofoo() {
// CHECK-NEXT: return T();
// CHECK-NEXT: }
template <typename T>
T foofoo() { return T(); }

// CHECK:      template<> int foofoo<int>() {
// CHECK-NEXT: return int();
// CHECK-NEXT: }

// CHECK:      #pragma omp declare variant(foofoo<int>) match(implementation={vendor(score(5):ibm)},device={kind(fpga)})
// CHECK-NEXT: #pragma omp declare variant(foofoo<int>) match(implementation={vendor(score(0):unknown)})
// CHECK-NEXT: #pragma omp declare variant(foofoo<int>) match(implementation={vendor(score(0):llvm)},device={kind(cpu)})
// CHECK-NEXT: int bar();
#pragma omp declare variant(foofoo <int>) match(xxx = {})
#pragma omp declare variant(foofoo <int>) match(xxx = {vvv})
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(llvm), xxx}, device={kind(cpu)})
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(unknown)})
#pragma omp declare variant(foofoo <int>) match(implementation={vendor(score(5): ibm)}, device={kind(fpga)})
int bar();

// CHECK:      #pragma omp declare variant(foofoo<T>) match(implementation={vendor(score(C + 5):ibm, xxx)},device={kind(cpu, host)})
// CHECK-NEXT: #pragma omp declare variant(foofoo<T>) match(implementation={vendor(score(0):unknown)})
// CHECK-NEXT: #pragma omp declare variant(foofoo<T>) match(implementation={vendor(score(0):llvm)},device={kind(cpu)})
// CHECK-NEXT: template <typename T, int C> T barbar();
#pragma omp declare variant(foofoo <T>) match(xxx = {})
#pragma omp declare variant(foofoo <T>) match(xxx = {vvv})
#pragma omp declare variant(foofoo <T>) match(user = {score(<expr>) : condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(user = {score(<expr>) : condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(user = {condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(user = {condition(<expr>)})
#pragma omp declare variant(foofoo <T>) match(implementation={vendor(llvm)},device={kind(cpu)})
#pragma omp declare variant(foofoo <T>) match(implementation={vendor(unknown)})
#pragma omp declare variant(foofoo <T>) match(implementation={vendor(score(C+5): ibm, xxx, ibm)},device={kind(cpu,host)})
template <typename T, int C>
T barbar();

// CHECK:      #pragma omp declare variant(foofoo<int>) match(implementation={vendor(score(3 + 5):ibm, xxx)},device={kind(cpu, host)})
// CHECK-NEXT: #pragma omp declare variant(foofoo<int>) match(implementation={vendor(score(0):unknown)})
// CHECK-NEXT: #pragma omp declare variant(foofoo<int>) match(implementation={vendor(score(0):llvm)},device={kind(cpu)})
// CHECK-NEXT: template<> int barbar<int, 3>();

// CHECK-NEXT: int baz() {
// CHECK-NEXT: return barbar<int, 3>();
// CHECK-NEXT: }
int baz() {
  return barbar<int, 3>();
}

// CHECK:      template <class C> void h_ref(C *hp, C *hp2, C *hq, C *lin) {
// CHECK-NEXT: }
// CHECK-NEXT: template<> void h_ref<double>(double *hp, double *hp2, double *hq, double *lin) {
// CHECK-NEXT: }
// CHECK-NEXT: template<> void h_ref<float>(float *hp, float *hp2, float *hq, float *lin) {
// CHECK-NEXT: }
template <class C>
void h_ref(C *hp, C *hp2, C *hq, C *lin) {
}

// CHECK:      #pragma omp declare variant(h_ref<C>) match(implementation={vendor(score(0):unknown)},device={kind(nohost)})
// CHECK-NEXT: #pragma omp declare variant(h_ref<C>) match(implementation={vendor(score(0):llvm)},device={kind(gpu)})
// CHECK-NEXT: template <class C> void h(C *hp, C *hp2, C *hq, C *lin) {
// CHECK-NEXT: }
#pragma omp declare variant(h_ref <C>) match(xxx = {})
#pragma omp declare variant(h_ref <C>) match(implementation={vendor(llvm)}, device={kind(gpu)})
#pragma omp declare variant(h_ref <C>) match(implementation={vendor(unknown)},device={kind(nohost)})
template <class C>
void h(C *hp, C *hp2, C *hq, C *lin) {
}

// CHECK:      #pragma omp declare variant(h_ref<float>) match(implementation={vendor(score(0):unknown)},device={kind(nohost)})
// CHECK-NEXT: #pragma omp declare variant(h_ref<float>) match(implementation={vendor(score(0):llvm)},device={kind(gpu)})
// CHECK-NEXT: template<> void h<float>(float *hp, float *hp2, float *hq, float *lin) {
// CHECK-NEXT: }

// CHECK-NEXT: template<> void h<double>(double *hp, double *hp2, double *hq, double *lin) {
// CHECK-NEXT:   h((float *)hp, (float *)hp2, (float *)hq, (float *)lin);
// CHECK-NEXT: }
#pragma omp declare variant(h_ref <double>) match(xxx = {})
#pragma omp declare variant(h_ref <double>) match(implementation={vendor(ibm)},device={kind(cpu,gpu)})
#pragma omp declare variant(h_ref <double>) match(implementation={vendor(unknown)})
template <>
void h(double *hp, double *hp2, double *hq, double *lin) {
  h((float *)hp, (float *)hp2, (float *)hq, (float *)lin);
}

// CHECK: int fn();
int fn();
// CHECK: int fn(int);
int fn(int);
// CHECK:      #pragma omp declare variant(fn) match(implementation={vendor(score(0):unknown)},device={kind(cpu, gpu)})
// CHECK-NEXT: #pragma omp declare variant(fn) match(implementation={vendor(score(0):llvm)})
// CHECK-NEXT: int overload();
#pragma omp declare variant(fn) match(xxx = {})
#pragma omp declare variant(fn) match(implementation={vendor(llvm)})
#pragma omp declare variant(fn) match(implementation={vendor(unknown)},device={kind(cpu,gpu)})
int overload(void);

// CHECK:      int fn_deduced_variant() {
// CHECK-NEXT: return 0;
// CHECK-NEXT: }
auto fn_deduced_variant() { return 0; }
// CHECK:      #pragma omp declare variant(fn_deduced_variant) match(implementation={vendor(score(0):unknown)},device={kind(gpu, nohost)})
// CHECK-NEXT: #pragma omp declare variant(fn_deduced_variant) match(implementation={vendor(score(0):llvm)},device={kind(cpu, host)})
// CHECK-NEXT: int fn_deduced();
#pragma omp declare variant(fn_deduced_variant) match(xxx = {})
#pragma omp declare variant(fn_deduced_variant) match(implementation={vendor(llvm)},device={kind(cpu,host)})
#pragma omp declare variant(fn_deduced_variant) match(implementation={vendor(unknown)},device={kind(gpu,nohost)})
int fn_deduced();

// CHECK: int fn_deduced_variant1();
int fn_deduced_variant1();
// CHECK:      #pragma omp declare variant(fn_deduced_variant1) match(implementation={vendor(score(0):unknown)},device={kind(cpu, host)})
// CHECK-NEXT: #pragma omp declare variant(fn_deduced_variant1) match(implementation={vendor(score(0):ibm)},device={kind(gpu, nohost)})
// CHECK-NEXT: int fn_deduced1() {
// CHECK-NEXT: return 0;
// CHECK-NEXT: }
#pragma omp declare variant(fn_deduced_variant1) match(xxx = {})
#pragma omp declare variant(fn_deduced_variant1) match(implementation={vendor(ibm)},device={kind(gpu,nohost)})
#pragma omp declare variant(fn_deduced_variant1) match(implementation={vendor(unknown)},device={kind(cpu,host)})
auto fn_deduced1() { return 0; }

// CHECK:      struct SpecialFuncs {
// CHECK-NEXT: void vd() {
// CHECK-NEXT: }
// CHECK-NEXT: SpecialFuncs();
// CHECK-NEXT: ~SpecialFuncs() noexcept;
// CHECK-NEXT: void baz() {
// CHECK-NEXT: }
// CHECK-NEXT: void bar() {
// CHECK-NEXT: }
// CHECK-NEXT: void bar(int) {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp declare variant(SpecialFuncs::baz) match(implementation={vendor(score(0):unknown)},device={kind(nohost)})
// CHECK-NEXT: #pragma omp declare variant(SpecialFuncs::bar) match(implementation={vendor(score(0):ibm)},device={kind(cpu)})
// CHECK-NEXT: void foo1() {
// CHECK-NEXT: }
// CHECK-NEXT: #pragma omp declare variant(SpecialFuncs::baz) match(implementation={vendor(score(0):unknown)},device={kind(cpu, host)})
// CHECK-NEXT: void xxx();
// CHECK-NEXT: } s;
struct SpecialFuncs {
  void vd() {}
  SpecialFuncs();
  ~SpecialFuncs();

  void baz() {}
  void bar() {}
  void bar(int) {}
#pragma omp declare variant(SpecialFuncs::baz) match(xxx = {})
#pragma omp declare variant(SpecialFuncs::bar) match(xxx = {})
#pragma omp declare variant(SpecialFuncs::bar) match(implementation={vendor(ibm)},device={kind(cpu)})
#pragma omp declare variant(SpecialFuncs::baz) match(implementation={vendor(unknown)},device={kind(nohost)})
  void foo1() {}
#pragma omp declare variant(SpecialFuncs::baz) match(implementation={vendor(unknown)},device={kind(cpu, host)})
  void xxx();
} s;

// CHECK:      #pragma omp declare variant(SpecialFuncs::baz) match(implementation={vendor(score(0):unknown)},device={kind(cpu, host)})
// CHECK-NEXT: void SpecialFuncs::xxx() {
// CHECK-NEXT: }
void SpecialFuncs::xxx() {}

// CHECK:      static void static_f_variant() {
// CHECK-NEXT: }
static void static_f_variant() {}
// CHECK:      #pragma omp declare variant(static_f_variant) match(implementation={vendor(score(0):unknown)})
// CHECK-NEXT: #pragma omp declare variant(static_f_variant) match(implementation={vendor(score(0):llvm)},device={kind(fpga)})
// CHECK-NEXT: static void static_f() {
// CHECK-NEXT: }
#pragma omp declare variant(static_f_variant) match(xxx = {})
#pragma omp declare variant(static_f_variant) match(implementation={vendor(llvm)},device={kind(fpga)})
#pragma omp declare variant(static_f_variant) match(implementation={vendor(unknown)})
static void static_f() {}

// CHECK: void bazzzz() {
// CHECK-NEXT: s.foo1();
// CHECK-NEXT: static_f();
// CHECK-NEXT: }
void bazzzz() {
  s.foo1();
  static_f();
}

// CHECK: int fn_linkage_variant();
// CHECK: extern "C" {
// CHECK:     #pragma omp declare variant(fn_linkage_variant) match(implementation={vendor(score(0):xxx)},device={kind(cpu, host)})
// CHECK:     int fn_linkage();
// CHECK: }
int fn_linkage_variant();
extern "C" {
#pragma omp declare variant(fn_linkage_variant) match(implementation = {vendor(xxx)},device={kind(cpu,host)})
int fn_linkage();
}

// CHECK: extern "C" int fn_linkage_variant1()
// CHECK: #pragma omp declare variant(fn_linkage_variant1) match(implementation={vendor(score(0):xxx)},device={kind(cpu, host)})
// CHECK: int fn_linkage1();
extern "C" int fn_linkage_variant1();
#pragma omp declare variant(fn_linkage_variant1) match(implementation = {vendor(xxx)},device={kind(cpu,host)})
int fn_linkage1();

