// RUN: %clang_cc1 -verify -fopenmp -triple x86_64-unknown-linux-gnu -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -verify -fopenmp-simd -triple x86_64-unknown-linux-gnu -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=DEFAULT

// RUN: %clang_cc1 -verify -fopenmp -triple amdgcn-amd-amdhsa -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=DEFAULT-AMDGCN

// RUN: %clang_cc1 -verify -fopenmp-simd -triple amdgcn-amd-amdhsa -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=DEFAULT-AMDGCN

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -DOMP52 -triple x86_64-unknown-linux-gnu -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=OMP52

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -DOMP52 -triple x86_64-unknown-linux-gnu -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=OMP52

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 -DOMP52 -triple amdgcn-amd-amdhsa -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=OMP52-AMDGCN

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=52 -DOMP52 -triple amdgcn-amd-amdhsa -x c -std=c99 -ast-print %s -o - | FileCheck %s --check-prefix=OMP52-AMDGCN
// expected-no-diagnostics

#ifndef HEADER
#define HEADER

#ifdef OMP52
void bar(void);

#define N 10
void foo1(void) {
#pragma omp metadirective when(device = {kind(cpu)} \
                               : parallel) otherwise()
  bar();
#pragma omp metadirective when(implementation = {vendor(score(0)  \
                                                        : llvm)}, \
                               device = {kind(cpu)}               \
                               : parallel) otherwise(target teams)
  bar();
#pragma omp metadirective when(device = {kind(gpu)}                                 \
                               : target teams) when(implementation = {vendor(llvm)} \
                                                    : parallel) otherwise()
  bar();
#pragma omp metadirective otherwise(target) when(implementation = {vendor(score(5)  \
                                                                        : llvm)}, \
                                               device = {kind(cpu, host)}         \
                                               : parallel)
  bar();
#pragma omp metadirective when(user = {condition(N > 10)}                 \
                               : target) when(user = {condition(N == 10)} \
                                              : parallel)
  bar();
#pragma omp metadirective when(device = {kind(host)} \
                               : parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_all)} \
                               : parallel) otherwise(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_any)} \
                               : parallel) otherwise(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_none)} \
                               : parallel) otherwise(parallel for)
  for (int i = 0; i < 100; i++)
    ;

// Test metadirective with nested OpenMP directive.
  int array[16];
  #pragma omp metadirective when(user = {condition(1)} \
                                 : parallel for)
  for (int i = 0; i < 16; i++) {
    #pragma omp simd
    for (int j = 0; j < 16; j++)
      array[i] = i;
  }

#pragma omp metadirective when(device={arch("amdgcn")}: \
                                teams distribute parallel for)\
                                otherwise(parallel for)
  for (int i = 0; i < 100; i++)
  ;

#pragma omp metadirective when(implementation = {extension(match_all)} \
                               : nothing) otherwise(parallel for)
  for (int i = 0; i < 16; i++)
    ;

#pragma omp metadirective when(implementation = {extension(match_any)} \
                               : parallel) otherwise(nothing)
  for (int i = 0; i < 16; i++)
    ;


#pragma omp metadirective when(user = {condition(0)}	\
			       : parallel for)
  for (int i=0; i<10; i++)
    ;
#pragma omp metadirective when(user = {condition(0)}		  \
			       : parallel for) when(implementation = {extension(match_none)} \
						    : parallel) otherwise(parallel for)
  for (int i=0; i<10; i++)
    ;


#pragma omp metadirective when(user = {condition(1)}	\
			       : parallel for)
  for (int i=0; i<10; i++)
    ;
#pragma omp metadirective when(user = {condition(1)}		  \
			       : parallel for) when(implementation = {extension(match_none)} \
						    : parallel) otherwise(parallel for)
  for (int i=0; i<10; i++)
    ;
}

// OMP52: void bar(void);
// OMP52: void foo1(void)
// OMP52-NEXT: #pragma omp parallel
// OMP52-NEXT: bar()
// OMP52-NEXT: #pragma omp parallel
// OMP52-NEXT: bar()
// OMP52-NEXT: #pragma omp parallel
// OMP52-NEXT: bar()
// OMP52-NEXT: #pragma omp parallel
// OMP52-NEXT: bar()
// OMP52-NEXT: #pragma omp parallel
// OMP52-NEXT: bar()
// OMP52-NEXT: #pragma omp parallel for
// OMP52-NEXT: for (int i = 0; i < 100; i++)
// OMP52: #pragma omp parallel
// OMP52-NEXT: for (int i = 0; i < 100; i++)
// OMP52: #pragma omp parallel for
// OMP52-NEXT: for (int i = 0; i < 100; i++)
// OMP52: #pragma omp parallel
// OMP52-NEXT: for (int i = 0; i < 100; i++)
// OMP52: #pragma omp parallel for
// OMP52-NEXT: for (int i = 0; i < 16; i++) {
// OMP52-NEXT: #pragma omp simd
// OMP52-NEXT: for (int j = 0; j < 16; j++)
// OMP52-AMDGCN: #pragma omp teams distribute parallel for
// OMP52-AMDGCN-NEXT: for (int i = 0; i < 100; i++)
// OMP52: for (int i = 0; i < 16; i++)
// OMP52: for (int i = 0; i < 16; i++)

#else
void bar(void);

#define N 10
void foo2(void) {
#pragma omp metadirective when(device = {kind(cpu)} \
                               : parallel) default()
  bar();
#pragma omp metadirective when(implementation = {vendor(score(0)  \
                                                        : llvm)}, \
                               device = {kind(cpu)}               \
                               : parallel) default(target teams)
  bar();
#pragma omp metadirective when(device = {kind(gpu)}                                 \
                               : target teams) when(implementation = {vendor(llvm)} \
                                                    : parallel) default()
  bar();
#pragma omp metadirective default(target) when(implementation = {vendor(score(5)  \
                                                                        : llvm)}, \
                                               device = {kind(cpu, host)}         \
                                               : parallel)
  bar();
#pragma omp metadirective when(user = {condition(N > 10)}                 \
                               : target) when(user = {condition(N == 10)} \
                                              : parallel)
  bar();
#pragma omp metadirective when(device = {kind(host)} \
                               : parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_all)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_any)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;
#pragma omp metadirective when(implementation = {extension(match_none)} \
                               : parallel) default(parallel for)
  for (int i = 0; i < 100; i++)
    ;

// Test metadirective with nested OpenMP directive.
  int array[16];
  #pragma omp metadirective when(user = {condition(1)} \
                                 : parallel for)
  for (int i = 0; i < 16; i++) {
    #pragma omp simd
    for (int j = 0; j < 16; j++)
      array[i] = i;
  }

#pragma omp metadirective when(device={arch("amdgcn")}: \
                                teams distribute parallel for)\
                                default(parallel for)
  for (int i = 0; i < 100; i++)
  ;

#pragma omp metadirective when(implementation = {extension(match_all)} \
                               : nothing) default(parallel for)
  for (int i = 0; i < 16; i++)
    ;

#pragma omp metadirective when(implementation = {extension(match_any)} \
                               : parallel) default(nothing)
  for (int i = 0; i < 16; i++)
    ;


#pragma omp metadirective when(user = {condition(0)}	\
			       : parallel for)
  for (int i=0; i<10; i++)
    ;
#pragma omp metadirective when(user = {condition(0)}		  \
			       : parallel for) when(implementation = {extension(match_none)} \
						    : parallel) default(parallel for)
  for (int i=0; i<10; i++)
    ;


#pragma omp metadirective when(user = {condition(1)}	\
			       : parallel for)
  for (int i=0; i<10; i++)
    ;
#pragma omp metadirective when(user = {condition(1)}		  \
			       : parallel for) when(implementation = {extension(match_none)} \
						    : parallel) default(parallel for)
  for (int i=0; i<10; i++)
    ;
#if _OPENMP >= 202111
    #pragma omp metadirective when(user = {condition(0)}	\
                 : parallel for) otherwise()
      for (int i=0; i<10; i++)
        ;
    
    #pragma omp metadirective when(user = {condition(1)}	\
                : parallel for) otherwise()
      for (int i=0; i<10; i++)
        ;
#endif
}

// DEFAULT: void bar(void);
// DEFAULT: void foo2(void)
// DEFAULT-NEXT: #pragma omp parallel
// DEFAULT-NEXT: bar()
// DEFAULT-NEXT: #pragma omp parallel
// DEFAULT-NEXT: bar()
// DEFAULT-NEXT: #pragma omp parallel
// DEFAULT-NEXT: bar()
// DEFAULT-NEXT: #pragma omp parallel
// DEFAULT-NEXT: bar()
// DEFAULT-NEXT: #pragma omp parallel
// DEFAULT-NEXT: bar()
// DEFAULT-NEXT: #pragma omp parallel for
// DEFAULT-NEXT: for (int i = 0; i < 100; i++)
// DEFAULT: #pragma omp parallel
// DEFAULT-NEXT: for (int i = 0; i < 100; i++)
// DEFAULT: #pragma omp parallel for
// DEFAULT-NEXT: for (int i = 0; i < 100; i++)
// DEFAULT: #pragma omp parallel
// DEFAULT-NEXT: for (int i = 0; i < 100; i++)
// DEFAULT: #pragma omp parallel for
// DEFAULT-NEXT: for (int i = 0; i < 16; i++) {
// DEFAULT-NEXT: #pragma omp simd
// DEFAULT-NEXT: for (int j = 0; j < 16; j++)
// DEFAULT-AMDGCN: #pragma omp teams distribute parallel for
// DEFAULT-AMDGCN-NEXT: for (int i = 0; i < 100; i++)
// DEFAULT: for (int i = 0; i < 16; i++)
// DEFAULT: for (int i = 0; i < 16; i++)


#endif
#endif

