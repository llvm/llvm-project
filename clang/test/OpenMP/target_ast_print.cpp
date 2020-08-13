

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// RUN: %clang_cc1 -DOMP45 -verify -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s --check-prefix=OMP45
// RUN: %clang_cc1 -DOMP45 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP45 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix=OMP45

// RUN: %clang_cc1 -DOMP45 -verify -fopenmp-simd -fopenmp-version=45 -ast-print %s | FileCheck %s --check-prefix=OMP45
// RUN: %clang_cc1 -DOMP45 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP45 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix=OMP45
#ifdef OMP45

void foo() {}

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i, j, a[20], always, close;
#pragma omp target
  foo();
#pragma omp target if (target:argc > 0)
  foo();
#pragma omp target if (C)
  foo();
#pragma omp target map(i)
  foo();
#pragma omp target map(a[0:10], i)
  foo();
#pragma omp target map(to: i) map(from: j)
  foo();
#pragma omp target map(always,alloc: i)
  foo();
#pragma omp target map(always from: i)
  foo();
#pragma omp target map(always)
  {always++;}
#pragma omp target map(always,i)
  {always++;i++;}
#pragma omp target map(close,alloc: i)
  foo();
#pragma omp target map(close from: i)
  foo();
#pragma omp target map(close)
  {close++;}
#pragma omp target map(close,i)
  {close++;i++;}
#pragma omp target nowait
  foo();
#pragma omp target depend(in : argc, argv[i:argc], a[:])
  foo();
#pragma omp target defaultmap(tofrom: scalar)
  foo();
  return 0;
}

// OMP45: template <typename T, int C> T tmain(T argc, T *argv) {
// OMP45-NEXT: T i, j, a[20]
// OMP45-NEXT: #pragma omp target{{$}}
// OMP45-NEXT: foo();
// OMP45-NEXT: #pragma omp target if(target: argc > 0)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target if(C)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: a[0:10],i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(to: i) map(from: j)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(always,alloc: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(always,from: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: always)
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(tofrom: always,i)
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(close,alloc: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(close,from: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: close)
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(tofrom: close,i)
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target nowait
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target defaultmap(tofrom: scalar)
// OMP45-NEXT: foo()
// OMP45: template<> int tmain<int, 5>(int argc, int *argv) {
// OMP45-NEXT: int i, j, a[20]
// OMP45-NEXT: #pragma omp target
// OMP45-NEXT: foo();
// OMP45-NEXT: #pragma omp target if(target: argc > 0)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target if(5)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: a[0:10],i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(to: i) map(from: j)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(always,alloc: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(always,from: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: always)
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(tofrom: always,i)
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(close,alloc: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(close,from: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: close)
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(tofrom: close,i)
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target nowait
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target defaultmap(tofrom: scalar)
// OMP45-NEXT: foo()
// OMP45: template<> char tmain<char, 1>(char argc, char *argv) {
// OMP45-NEXT: char i, j, a[20]
// OMP45-NEXT: #pragma omp target
// OMP45-NEXT: foo();
// OMP45-NEXT: #pragma omp target if(target: argc > 0)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target if(1)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: a[0:10],i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(to: i) map(from: j)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(always,alloc: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(always,from: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: always)
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(tofrom: always,i)
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(close,alloc: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(close,from: i)
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target map(tofrom: close)
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target map(tofrom: close,i)
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }
// OMP45-NEXT: #pragma omp target nowait
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
// OMP45-NEXT: foo()
// OMP45-NEXT: #pragma omp target defaultmap(tofrom: scalar)
// OMP45-NEXT: foo()

// OMP45-LABEL: class S {
class S {
  void foo() {
// OMP45-NEXT: void foo() {
    int a = 0;
// OMP45-NEXT: int a = 0;
    #pragma omp target map(this[0])
// OMP45-NEXT: #pragma omp target map(tofrom: this[0])
      a++;
// OMP45-NEXT: a++;
    #pragma omp target map(this[:1])
// OMP45-NEXT: #pragma omp target map(tofrom: this[:1])
      a++;
// OMP45-NEXT: a++;
    #pragma omp target map((this)[0])
// OMP45-NEXT: #pragma omp target map(tofrom: (this)[0])
      a++;
// OMP45-NEXT: a++;
    #pragma omp target map(this[:a])
// OMP45-NEXT: #pragma omp target map(tofrom: this[:a])
      a++;
// OMP45-NEXT: a++;
    #pragma omp target map(this[a:1])
// OMP45-NEXT: #pragma omp target map(tofrom: this[a:1])
      a++;
// OMP45-NEXT: a++;
    #pragma omp target map(this[a])
// OMP45-NEXT: #pragma omp target map(tofrom: this[a])
      a++;
// OMP45-NEXT: a++;
  }
// OMP45-NEXT: }
};
// OMP45-NEXT: };

// OMP45-LABEL: int main(int argc, char **argv) {
int main (int argc, char **argv) {
  int i, j, a[20], always, close;
// OMP45-NEXT: int i, j, a[20]
#pragma omp target
// OMP45-NEXT: #pragma omp target
  foo();
// OMP45-NEXT: foo();
#pragma omp target if (argc > 0)
// OMP45-NEXT: #pragma omp target if(argc > 0)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(i) if(argc>0)
// OMP45-NEXT: #pragma omp target map(tofrom: i) if(argc > 0)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(i)
// OMP45-NEXT: #pragma omp target map(tofrom: i)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(a[0:10], i)
// OMP45-NEXT: #pragma omp target map(tofrom: a[0:10],i)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(to: i) map(from: j)
// OMP45-NEXT: #pragma omp target map(to: i) map(from: j)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(always,alloc: i)
// OMP45-NEXT: #pragma omp target map(always,alloc: i)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(always from: i)
// OMP45-NEXT: #pragma omp target map(always,from: i)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(always)
// OMP45-NEXT: #pragma omp target map(tofrom: always)
  {always++;}
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: }

#pragma omp target map(always,i)
// OMP45-NEXT: #pragma omp target map(tofrom: always,i)
  {always++;i++;}
// OMP45-NEXT: {
// OMP45-NEXT: always++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }

#pragma omp target map(close,alloc: i)
// OMP45-NEXT: #pragma omp target map(close,alloc: i)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(close from: i)
// OMP45-NEXT: #pragma omp target map(close,from: i)
  foo();
// OMP45-NEXT: foo();

#pragma omp target map(close)
// OMP45-NEXT: #pragma omp target map(tofrom: close)
  {close++;}
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: }

#pragma omp target map(close,i)
// OMP45-NEXT: #pragma omp target map(tofrom: close,i)
  {close++;i++;}
// OMP45-NEXT: {
// OMP45-NEXT: close++;
// OMP45-NEXT: i++;
// OMP45-NEXT: }

#pragma omp target nowait
// OMP45-NEXT: #pragma omp target nowait
  foo();
// OMP45-NEXT: foo();

#pragma omp target depend(in : argc, argv[i:argc], a[:])
// OMP45-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
  foo();
// OMP45-NEXT: foo();

#pragma omp target defaultmap(tofrom: scalar)
// OMP45-NEXT: #pragma omp target defaultmap(tofrom: scalar)
  foo();
// OMP45-NEXT: foo();

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif

#ifdef OMP5

///==========================================================================///
// RUN: %clang_cc1 -DOMP5 -verify -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s --check-prefix OMP5
// RUN: %clang_cc1 -DOMP5 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP5 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix OMP5

// RUN: %clang_cc1 -DOMP5 -verify -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s --check-prefix OMP5
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -fsyntax-only -verify %s -ast-print | FileCheck %s --check-prefix OMP5

typedef void **omp_allocator_handle_t;
extern const omp_allocator_handle_t omp_null_allocator;
extern const omp_allocator_handle_t omp_default_mem_alloc;
extern const omp_allocator_handle_t omp_large_cap_mem_alloc;
extern const omp_allocator_handle_t omp_const_mem_alloc;
extern const omp_allocator_handle_t omp_high_bw_mem_alloc;
extern const omp_allocator_handle_t omp_low_lat_mem_alloc;
extern const omp_allocator_handle_t omp_cgroup_mem_alloc;
extern const omp_allocator_handle_t omp_pteam_mem_alloc;
extern const omp_allocator_handle_t omp_thread_mem_alloc;

void foo() {}

#pragma omp declare target
void bar() {}
#pragma omp end declare target

int a;
#pragma omp declare target link(a)

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i, j, a[20], always, close;
#pragma omp target device(argc)
  foo();
#pragma omp target if (target:argc > 0) device(device_num: C)
  foo();
#pragma omp target if (C) device(ancestor: argc)
  foo();
#pragma omp target map(i)
  foo();
#pragma omp target map(a[0:10], i)
  foo();
#pragma omp target map(to: i) map(from: j)
  foo();
#pragma omp target map(always,alloc: i)
  foo();
#pragma omp target map(always from: i)
  foo();
#pragma omp target map(always)
  {always++;}
#pragma omp target map(always,i)
  {always++;i++;}
#pragma omp target map(close,alloc: i)
  foo();
#pragma omp target map(close from: i)
  foo();
#pragma omp target map(close)
  {close++;}
#pragma omp target map(close,i)
  {close++;i++;}
#pragma omp target nowait
  foo();
#pragma omp target depend(in : argc, argv[i:argc], a[:])
  foo();
#pragma omp target defaultmap(alloc: scalar)
  foo();
#pragma omp target defaultmap(to: scalar)
  foo();
#pragma omp target defaultmap(from: scalar)
  foo();
#pragma omp target defaultmap(tofrom: scalar)
  foo();
#pragma omp target defaultmap(firstprivate: scalar)
  foo();
#pragma omp target defaultmap(none: scalar)
  foo();
#pragma omp target defaultmap(default: scalar)
  foo();
#pragma omp target defaultmap(alloc: aggregate)
  foo();
#pragma omp target defaultmap(to: aggregate)
  foo();
#pragma omp target defaultmap(from: aggregate)
  foo();
#pragma omp target defaultmap(tofrom: aggregate)
  foo();
#pragma omp target defaultmap(firstprivate: aggregate)
  foo();
#pragma omp target defaultmap(none: aggregate)
  foo();
#pragma omp target defaultmap(default: aggregate)
  foo();
#pragma omp target defaultmap(alloc: pointer)
  foo();
#pragma omp target defaultmap(to: pointer)
  foo();
#pragma omp target defaultmap(from: pointer)
  foo();
#pragma omp target defaultmap(tofrom: pointer)
  foo();
#pragma omp target defaultmap(firstprivate: pointer)
  foo();
#pragma omp target defaultmap(none: pointer)
  foo();
#pragma omp target defaultmap(default: pointer)
  foo();
#pragma omp target defaultmap(to: scalar) defaultmap(tofrom: pointer)
  foo();
#pragma omp target defaultmap(from: pointer) defaultmap(none: aggregate)
  foo();
#pragma omp target defaultmap(default: aggregate) defaultmap(alloc: scalar)
  foo();
#pragma omp target defaultmap(alloc: aggregate) defaultmap(firstprivate: scalar) defaultmap(tofrom: pointer)
  foo();
#pragma omp target defaultmap(tofrom: aggregate) defaultmap(to: pointer) defaultmap(alloc: scalar)
  foo();

  int *g;

#pragma omp target is_device_ptr(g) defaultmap(none: pointer)
  g++;
#pragma omp target private(g) defaultmap(none: pointer)
  g++;
#pragma omp target firstprivate(g) defaultmap(none: pointer)
  g++;
#pragma omp target defaultmap(none: scalar) map(to: i)
  i++;
#pragma omp target defaultmap(none: aggregate) map(to: a)
  a[3]++;
#pragma omp target defaultmap(none: scalar)
  bar();

  return 0;
}

// OMP5: template <typename T, int C> T tmain(T argc, T *argv) {
// OMP5-NEXT: T i, j, a[20]
// OMP5-NEXT: #pragma omp target device(argc){{$}}
// OMP5-NEXT: foo();
// OMP5-NEXT: #pragma omp target if(target: argc > 0) device(device_num: C)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target if(C) device(ancestor: argc)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: a[0:10],i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(to: i) map(from: j)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(always,alloc: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(always,from: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: always)
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(tofrom: always,i)
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(close,alloc: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(close,from: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: close)
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(tofrom: close,i)
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target nowait
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer) defaultmap(none: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate) defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate) defaultmap(firstprivate: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate) defaultmap(to: pointer) defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5: template<> int tmain<int, 5>(int argc, int *argv) {
// OMP5-NEXT: int i, j, a[20]
// OMP5-NEXT: #pragma omp target
// OMP5-NEXT: foo();
// OMP5-NEXT: #pragma omp target if(target: argc > 0)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target if(5)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: a[0:10],i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(to: i) map(from: j)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(always,alloc: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(always,from: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: always)
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(tofrom: always,i)
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(close,alloc: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(close,from: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: close)
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(tofrom: close,i)
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target nowait
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer) defaultmap(none: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate) defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate) defaultmap(firstprivate: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate) defaultmap(to: pointer) defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5: template<> char tmain<char, 1>(char argc, char *argv) {
// OMP5-NEXT: char i, j, a[20]
// OMP5-NEXT: #pragma omp target device(argc)
// OMP5-NEXT: foo();
// OMP5-NEXT: #pragma omp target if(target: argc > 0) device(device_num: 1)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target if(1) device(ancestor: argc)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: a[0:10],i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(to: i) map(from: j)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(always,alloc: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(always,from: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: always)
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(tofrom: always,i)
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(close,alloc: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(close,from: i)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target map(tofrom: close)
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target map(tofrom: close,i)
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }
// OMP5-NEXT: #pragma omp target nowait
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(none: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer) defaultmap(none: aggregate)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate) defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate) defaultmap(firstprivate: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: foo()
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate) defaultmap(to: pointer) defaultmap(alloc: scalar)
// OMP5-NEXT: foo()
// OMP5-NEXT: int *g;
// OMP5-NEXT: #pragma omp target is_device_ptr(g) defaultmap(none: pointer)
// OMP5-NEXT: g++;
// OMP5-NEXT: #pragma omp target private(g) defaultmap(none: pointer)
// OMP5-NEXT: g++;
// OMP5-NEXT: #pragma omp target firstprivate(g) defaultmap(none: pointer)
// OMP5-NEXT: g++;
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar) map(to: i)
// OMP5-NEXT: i++;
// OMP5-NEXT: #pragma omp target defaultmap(none: aggregate) map(to: a)
// OMP5-NEXT: a[3]++;
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar)
// OMP5-NEXT: bar();

// OMP5-LABEL: class S {
class S {
  void foo() {
// OMP5-NEXT: void foo() {
    int a = 0;
// OMP5-NEXT: int a = 0;
    #pragma omp target map(this[0])
// OMP5-NEXT: #pragma omp target map(tofrom: this[0])
      a++;
// OMP5-NEXT: a++;
    #pragma omp target map(this[:1])
// OMP5-NEXT: #pragma omp target map(tofrom: this[:1])
      a++;
// OMP5-NEXT: a++;
    #pragma omp target map((this)[0])
// OMP5-NEXT: #pragma omp target map(tofrom: (this)[0])
      a++;
// OMP5-NEXT: a++;
    #pragma omp target map(this[:a])
// OMP5-NEXT: #pragma omp target map(tofrom: this[:a])
      a++;
// OMP5-NEXT: a++;
    #pragma omp target map(this[a:1])
// OMP5-NEXT: #pragma omp target map(tofrom: this[a:1])
      a++;
// OMP5-NEXT: a++;
    #pragma omp target map(this[a])
// OMP5-NEXT: #pragma omp target map(tofrom: this[a])
      a++;
// OMP5-NEXT: a++;
  }
// OMP5-NEXT: }
};
// OMP5-NEXT: };

// OMP5-LABEL: int main(int argc, char **argv) {
int main (int argc, char **argv) {
  int i, j, a[20], always, close;
// OMP5-NEXT: int i, j, a[20]
#pragma omp target
// OMP5-NEXT: #pragma omp target
  foo();
// OMP5-NEXT: foo();
#pragma omp target if (argc > 0)
// OMP5-NEXT: #pragma omp target if(argc > 0)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(i) if(argc>0)
// OMP5-NEXT: #pragma omp target map(tofrom: i) if(argc > 0)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(i)
// OMP5-NEXT: #pragma omp target map(tofrom: i)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(a[0:10], i)
// OMP5-NEXT: #pragma omp target map(tofrom: a[0:10],i)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(to: i) map(from: j)
// OMP5-NEXT: #pragma omp target map(to: i) map(from: j)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(always,alloc: i)
// OMP5-NEXT: #pragma omp target map(always,alloc: i)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(always from: i)
// OMP5-NEXT: #pragma omp target map(always,from: i)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(always)
// OMP5-NEXT: #pragma omp target map(tofrom: always)
  {always++;}
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: }

#pragma omp target map(always,i)
// OMP5-NEXT: #pragma omp target map(tofrom: always,i)
  {always++;i++;}
// OMP5-NEXT: {
// OMP5-NEXT: always++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }

#pragma omp target map(close,alloc: i)
// OMP5-NEXT: #pragma omp target map(close,alloc: i)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(close from: i)
// OMP5-NEXT: #pragma omp target map(close,from: i)
  foo();
// OMP5-NEXT: foo();

#pragma omp target map(close)
// OMP5-NEXT: #pragma omp target map(tofrom: close)
  {close++;}
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: }

#pragma omp target map(close,i)
// OMP5-NEXT: #pragma omp target map(tofrom: close,i)
  {close++;i++;}
// OMP5-NEXT: {
// OMP5-NEXT: close++;
// OMP5-NEXT: i++;
// OMP5-NEXT: }

#pragma omp target nowait
// OMP5-NEXT: #pragma omp target nowait
  foo();
// OMP5-NEXT: foo();

#pragma omp target depend(in : argc, argv[i:argc], a[:])
// OMP5-NEXT: #pragma omp target depend(in : argc,argv[i:argc],a[:])
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(alloc: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(alloc: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(to: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(from: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(from: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(tofrom: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(firstprivate: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(none: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(default: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(default: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(alloc: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(to: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(to: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(from: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(from: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(tofrom: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(firstprivate: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(none: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(none: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(default: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(alloc: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(alloc: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(to: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(to: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(from: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(tofrom: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(firstprivate: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(firstprivate: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(none: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(none: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(default: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(default: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(to: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(to: scalar) defaultmap(tofrom: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(from: pointer) defaultmap(none: aggregate)
// OMP5-NEXT: #pragma omp target defaultmap(from: pointer) defaultmap(none: aggregate)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(default: aggregate) defaultmap(alloc: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(default: aggregate) defaultmap(alloc: scalar)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(alloc: aggregate) defaultmap(firstprivate: scalar) defaultmap(tofrom: pointer)
// OMP5-NEXT: #pragma omp target defaultmap(alloc: aggregate) defaultmap(firstprivate: scalar) defaultmap(tofrom: pointer)
  foo();
// OMP5-NEXT: foo();

#pragma omp target defaultmap(tofrom: aggregate) defaultmap(to: pointer) defaultmap(alloc: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(tofrom: aggregate) defaultmap(to: pointer) defaultmap(alloc: scalar)
  foo();
// OMP5-NEXT: foo();

  int *g;
// OMP5-NEXT: int *g;

#pragma omp target is_device_ptr(g) defaultmap(none: pointer)
// OMP5-NEXT: #pragma omp target is_device_ptr(g) defaultmap(none: pointer)
  g++;
// OMP5-NEXT: g++;

#pragma omp target private(g) defaultmap(none: pointer)
// OMP5-NEXT: #pragma omp target private(g) defaultmap(none: pointer)
  g++;
// OMP5-NEXT: g++;

#pragma omp target firstprivate(g) defaultmap(none: pointer)
// OMP5-NEXT: #pragma omp target firstprivate(g) defaultmap(none: pointer)
  g++;
// OMP5-NEXT: g++;

#pragma omp target defaultmap(none: scalar) map(to: i)
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar) map(to: i)
  i++;
// OMP5-NEXT: i++;

#pragma omp target defaultmap(none: aggregate) map(to: a)
// OMP5-NEXT: #pragma omp target defaultmap(none: aggregate) map(to: a)
  a[3]++;
// OMP5-NEXT: a[3]++;

#pragma omp target defaultmap(none: scalar)
// OMP5-NEXT: #pragma omp target defaultmap(none: scalar)
  bar();
// OMP5-NEXT: bar();
#pragma omp target defaultmap(none)
  // OMP5-NEXT: #pragma omp target defaultmap(none)
  // OMP5-NEXT: bar();
  bar();
#pragma omp target allocate(omp_default_mem_alloc:argv) uses_allocators(omp_default_mem_alloc,omp_large_cap_mem_alloc) allocate(omp_large_cap_mem_alloc:argc) private(argc, argv)
  // OMP5-NEXT: #pragma omp target allocate(omp_default_mem_alloc: argv) uses_allocators(omp_default_mem_alloc,omp_large_cap_mem_alloc) allocate(omp_large_cap_mem_alloc: argc) private(argc,argv)
  // OMP5-NEXT: bar();
  bar();
  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif //OMP5
#endif
