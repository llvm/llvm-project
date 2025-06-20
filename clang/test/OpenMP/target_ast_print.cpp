

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// RUN: %clang_cc1 -DOMP45 -verify -Wno-vla -fopenmp -fopenmp-version=45 -ast-print %s | FileCheck %s --check-prefix=OMP45
// RUN: %clang_cc1 -DOMP45 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP45 -fopenmp -fopenmp-version=45 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix=OMP45

// RUN: %clang_cc1 -DOMP45 -verify -Wno-vla -fopenmp-simd -fopenmp-version=45 -ast-print %s | FileCheck %s --check-prefix=OMP45
// RUN: %clang_cc1 -DOMP45 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP45 -fopenmp-simd -fopenmp-version=45 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix=OMP45
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
#pragma omp target map(close, from: i)
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
// RUN: %clang_cc1 -DOMP5 -verify -Wno-vla -fopenmp -fopenmp-version=50 -ast-print %s | FileCheck %s --check-prefix OMP5
// RUN: %clang_cc1 -DOMP5 -fopenmp -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP5 -fopenmp -fopenmp-version=50 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP5

// RUN: %clang_cc1 -DOMP5 -verify -Wno-vla -fopenmp-simd -fopenmp-version=50 -ast-print %s | FileCheck %s --check-prefix OMP5
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=50 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=50 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP5

// RUN: %clang_cc1 -DOMP5 -verify -Wno-vla -fopenmp -fopenmp-version=99 -DOMP99 -ast-print %s | FileCheck %s --check-prefixes=OMP5,REV
// RUN: %clang_cc1 -DOMP5 -fopenmp -fopenmp-version=99 -DOMP99 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP5 -fopenmp -fopenmp-version=99 -DOMP99 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefixes=OMP5,REV

// RUN: %clang_cc1 -DOMP5 -verify -Wno-vla -fopenmp-simd -fopenmp-version=99 -DOMP99 -ast-print %s | FileCheck %s --check-prefixes=OMP5,REV
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=99 -DOMP99 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP5 -fopenmp-simd -fopenmp-version=99 -DOMP99 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefixes=OMP5,REV

#ifdef OMP99
#pragma omp requires reverse_offload
#endif
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
#ifdef OMP99
#pragma omp target if (C) device(ancestor: argc)
  foo();
#endif
#pragma omp target map(i)
  foo();
#pragma omp target map(a[0:10], i)
  foo();
#pragma omp target map(to: i) map(from: j)
  foo();
#pragma omp target map(always,alloc: i)
  foo();
#pragma omp target map(always, from: i)
  foo();
#pragma omp target map(always)
  {always++;}
#pragma omp target map(always,i)
  {always++;i++;}
#pragma omp target map(close,alloc: i)
  foo();
#pragma omp target map(close, from: i)
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
// REV: #pragma omp target if(C) device(ancestor: argc)
// REV-NEXT: foo()
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
// REV: #pragma omp target if(5)
// REV-NEXT: foo()
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
// REV: #pragma omp target if(1) device(ancestor: argc)
// REV-NEXT: foo()
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

#pragma omp target map(always, from: i)
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

#pragma omp target map(close, from: i)
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

#ifdef OMP51

///==========================================================================///
// RUN: %clang_cc1 -DOMP51 -verify -Wno-vla -fopenmp -ast-print %s | FileCheck %s --check-prefix OMP51
// RUN: %clang_cc1 -DOMP51 -fopenmp -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP51

// RUN: %clang_cc1 -DOMP51 -verify -Wno-vla -fopenmp-simd -ast-print %s | FileCheck %s --check-prefix OMP51
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP51 -fopenmp-simd -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP51

void foo() {}

template <typename T, int C>
T tmain(T argc, T *argv) {
  #pragma omp target defaultmap(present: scalar)
  foo();
  #pragma omp target defaultmap(present: aggregate)
  foo();
  #pragma omp target defaultmap(present: pointer)
  foo();
  #pragma omp target thread_limit(C)
  foo();
  #pragma omp target defaultmap(present)
  foo();

  return 0;
}

// OMP51: template <typename T, int C> T tmain(T argc, T *argv) {
// OMP51-NEXT: #pragma omp target defaultmap(present: scalar)
// OMP51-NEXT: foo()
// OMP51-NEXT: #pragma omp target defaultmap(present: aggregate)
// OMP51-NEXT: foo()
// OMP51-NEXT: #pragma omp target defaultmap(present: pointer)
// OMP51-NEXT: foo()
// OMP51-NEXT: #pragma omp target thread_limit(C)
// OMP51-NEXT: foo()
// OMP51-NEXT: #pragma omp target defaultmap(present)
// OMP51-NEXT: foo()

// OMP51-LABEL: int main(int argc, char **argv) {
int main (int argc, char **argv) {
#pragma omp target defaultmap(present: scalar)
// OMP51-NEXT: #pragma omp target defaultmap(present: scalar)
  foo();
// OMP51-NEXT: foo();
#pragma omp target defaultmap(present: aggregate)
// OMP51-NEXT: #pragma omp target defaultmap(present: aggregate)
  foo();
// OMP51-NEXT: foo();
#pragma omp target defaultmap(present: pointer)
// OMP51-NEXT: #pragma omp target defaultmap(present: pointer)
  foo();
// OMP51-NEXT: foo();

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}
#endif // OMP51

#ifdef OMP52

///==========================================================================///
// RUN: %clang_cc1 -DOMP52 -verify -Wno-vla -fopenmp -fopenmp-version=52 -ast-print %s | FileCheck %s --check-prefix OMP52
// RUN: %clang_cc1 -DOMP52 -fopenmp -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP52 -fopenmp -fopenmp-version=52 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP52

// RUN: %clang_cc1 -DOMP52 -verify -Wno-vla -fopenmp-simd -fopenmp-version=52 -ast-print %s | FileCheck %s --check-prefix OMP52
// RUN: %clang_cc1 -DOMP52 -fopenmp-simd -fopenmp-version=52 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP52 -fopenmp-simd -fopenmp-version=52 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP52

void foo() {}

template <typename T, int C>
T tmain(T argc, T *argv) {
  int N = 100;
  int v[N];
  #pragma omp target map(iterator(it = 0:N:2), to: v[it])
  foo();
  #pragma omp target map(iterator(it = 0:N:4), from: v[it])
  foo();

  return 0;
}

// OMP52: template <typename T, int C> T tmain(T argc, T *argv) {
// OMP52-NEXT: int N = 100;
// OMP52-NEXT: int v[N];
// OMP52-NEXT: #pragma omp target map(iterator(int it = 0:N:2),to: v[it])
// OMP52-NEXT: foo()
// OMP52-NEXT: #pragma omp target map(iterator(int it = 0:N:4),from: v[it])
// OMP52-NEXT: foo()

// OMP52-LABEL: int main(int argc, char **argv) {
int main (int argc, char **argv) {
  int i, j, a[20], always, close;
// OMP52-NEXT: int i, j, a[20]
#pragma omp target
// OMP52-NEXT: #pragma omp target
  foo();
// OMP52-NEXT: foo();
#pragma omp target map(iterator(it = 0:20:2), to: a[it])
// OMP52-NEXT: #pragma omp target map(iterator(int it = 0:20:2),to: a[it])
  foo();
// OMP52-NEXT: foo();
#pragma omp target map(iterator(it = 0:20:4), from: a[it])
// OMP52-NEXT: #pragma omp target map(iterator(int it = 0:20:4),from: a[it])
foo();
// OMP52-NEXT: foo();

  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}
#endif // OMP52

#ifdef OMP60

///==========================================================================///
// RUN: %clang_cc1 -DOMP60 -verify -Wno-vla -fopenmp -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix OMP60
// RUN: %clang_cc1 -DOMP60 -fopenmp -fopenmp-version=60 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP60 -fopenmp -fopenmp-version=60 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP60

// RUN: %clang_cc1 -DOMP60 -verify -Wno-vla -fopenmp-simd -fopenmp-version=60 -ast-print %s | FileCheck %s --check-prefix OMP60
// RUN: %clang_cc1 -DOMP60 -fopenmp-simd -fopenmp-version=60 -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMP60 -fopenmp-simd -fopenmp-version=60 -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix OMP60

void foo() {}
template <typename T, int C>
T tmain(T argc, T *argv) {
  T i;
#pragma omp target map(from always: i)
  foo();
#pragma omp target map(from, close: i)
  foo();
#pragma omp target map(always,close: i)
  foo();
  return 0;
}
//OMP60: template <typename T, int C> T tmain(T argc, T *argv) {
//OMP60-NEXT: T i;
//OMP60-NEXT: #pragma omp target map(always,from: i)
//OMP60-NEXT:     foo();
//OMP60-NEXT: #pragma omp target map(close,from: i)
//OMP60-NEXT:     foo();
//OMP60-NEXT: #pragma omp target map(always,close,tofrom: i)
//OMP60-NEXT:     foo();
//OMP60-NEXT: return 0;
//OMP60-NEXT:}
//OMP60:  template<> int tmain<int, 5>(int argc, int *argv) {
//OMP60-NEXT:  int i;
//OMP60-NEXT:  #pragma omp target map(always,from: i)
//OMP60-NEXT:      foo();
//OMP60-NEXT:  #pragma omp target map(close,from: i)
//OMP60-NEXT:      foo();
//OMP60-NEXT:  #pragma omp target map(always,close,tofrom: i)
//OMP60-NEXT:      foo();
//OMP60-NEXT:  return 0;
//OMP60-NEXT:}
//OMP60:  template<> char tmain<char, 1>(char argc, char *argv) {
//OMP60-NEXT:  char i;
//OMP60-NEXT:  #pragma omp target map(always,from: i)
//OMP60-NEXT:      foo();
//OMP60-NEXT:  #pragma omp target map(close,from: i)
//OMP60-NEXT:      foo();
//OMP60-NEXT:  #pragma omp target map(always,close,tofrom: i)
//OMP60-NEXT:      foo();
//OMP60-NEXT:  return 0;
//OMP60-NEXT:}
int main (int argc, char **argv) {
  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}
#endif // OMP60

#ifdef OMPX

// RUN: %clang_cc1 -DOMPX -verify -Wno-vla -fopenmp -fopenmp-extensions -ast-print %s | FileCheck %s --check-prefix=OMPX
// RUN: %clang_cc1 -DOMPX -fopenmp -fopenmp-extensions -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMPX -fopenmp -fopenmp-extensions -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix=OMPX

// RUN: %clang_cc1 -DOMPX -verify -Wno-vla -fopenmp-simd -fopenmp-extensions -ast-print %s | FileCheck %s --check-prefix=OMPX
// RUN: %clang_cc1 -DOMPX -fopenmp-simd -fopenmp-extensions -x c++ -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -DOMPX -fopenmp-simd -fopenmp-extensions -std=c++11 -include-pch %t -verify -Wno-vla %s -ast-print | FileCheck %s --check-prefix=OMPX

void foo() {}

template <typename T, int C>
T tmain(T argc, T *argv) {
  T i, ompx_hold;
#pragma omp target map(ompx_hold,alloc: i)
  foo();
#pragma omp target map(ompx_hold from: i)
  foo();
#pragma omp target map(ompx_hold)
  {ompx_hold++;}
#pragma omp target map(ompx_hold,i)
  {ompx_hold++;i++;}
  return 0;
}

//      OMPX: template <typename T, int C> T tmain(T argc, T *argv) {
// OMPX-NEXT:   T i, ompx_hold;
// OMPX-NEXT:   #pragma omp target map(ompx_hold,alloc: i)
// OMPX-NEXT:   foo()
// OMPX-NEXT:   #pragma omp target map(ompx_hold,from: i)
// OMPX-NEXT:   foo()
// OMPX-NEXT:   #pragma omp target map(tofrom: ompx_hold)
// OMPX-NEXT:   {
// OMPX-NEXT:     ompx_hold++;
// OMPX-NEXT:   }
// OMPX-NEXT:   #pragma omp target map(tofrom: ompx_hold,i)
// OMPX-NEXT:   {
// OMPX-NEXT:     ompx_hold++;
// OMPX-NEXT:     i++;
// OMPX-NEXT:   }

// OMPX-LABEL: int main(int argc, char **argv) {
//  OMPX-NEXT:   int i, ompx_hold;
//  OMPX-NEXT:   #pragma omp target map(ompx_hold,alloc: i)
//  OMPX-NEXT:   foo();
//  OMPX-NEXT:   #pragma omp target map(ompx_hold,from: i)
//  OMPX-NEXT:   foo();
//  OMPX-NEXT:   #pragma omp target map(tofrom: ompx_hold)
//  OMPX-NEXT:   {
//  OMPX-NEXT:     ompx_hold++;
//  OMPX-NEXT:   }
//  OMPX-NEXT:   #pragma omp target map(tofrom: ompx_hold,i)
//  OMPX-NEXT:   {
//  OMPX-NEXT:     ompx_hold++;
//  OMPX-NEXT:     i++;
//  OMPX-NEXT:   }
int main (int argc, char **argv) {
  int i, ompx_hold;
  #pragma omp target map(ompx_hold,alloc: i)
  foo();
  #pragma omp target map(ompx_hold from: i)
  foo();
  #pragma omp target map(ompx_hold)
  {ompx_hold++;}
  #pragma omp target map(ompx_hold,i)
  {ompx_hold++;i++;}
  return tmain<int, 5>(argc, &argc) + tmain<char, 1>(argv[0][0], argv[0]);
}

#endif
#endif
