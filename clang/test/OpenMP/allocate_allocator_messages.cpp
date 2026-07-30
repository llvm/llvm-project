// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp -ferror-limit 100 -o - %s -Wuninitialized

// RUN: %clang_cc1 -triple x86_64-apple-macos10.7.0 -verify -fopenmp-simd -ferror-limit 100 -o - %s -Wuninitialized

struct St{
 int a;
};

int sss;
#pragma omp allocate(sss) allocat // expected-warning {{extra tokens at the end of '#pragma omp allocate' are ignored}}
#pragma omp allocate(sss) allocate(sss) // expected-error {{unexpected OpenMP clause 'allocate' in directive '#pragma omp allocate'}}
#pragma omp allocate(sss) allocator // expected-error {{expected '(' after 'allocator'}}
#pragma omp allocate(sss) allocator(0,  // expected-error {{expected ')'}} expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}} expected-note {{to match this '('}}
#pragma omp allocate(sss) allocator(0,sss  // expected-error {{expected ')'}} expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}} expected-note {{to match this '('}}
#pragma omp allocate(sss) allocator(0,sss)  // expected-error {{expected ')'}} expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}} expected-note {{to match this '('}}
#pragma omp allocate(sss) allocator(sss)  // expected-error {{'omp_allocator_handle_t' type not found; include <omp.h>}}

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

struct St1{
 int a;
 static int b;
#pragma omp allocate(b) allocator(sss) // expected-error {{incompatible integer to pointer conversion initializing 'const omp_allocator_handle_t' (aka 'void **const') with an expression of type 'int'}} expected-note {{previous allocator is specified here}}
#pragma omp allocate(b)
#pragma omp allocate(b) allocator(omp_thread_mem_alloc) // expected-warning {{allocate directive specifies 'omp_thread_mem_alloc' allocator while previously used default}}
} d; // expected-note 2 {{'d' defined here}}

// expected-error@+1 {{expected one of the predefined allocators for the variables with the static storage: 'omp_default_mem_alloc', 'omp_large_cap_mem_alloc', 'omp_const_mem_alloc', 'omp_high_bw_mem_alloc', 'omp_low_lat_mem_alloc', 'omp_cgroup_mem_alloc', 'omp_pteam_mem_alloc' or 'omp_thread_mem_alloc'}}
#pragma omp allocate(d) allocator(nullptr)
extern void **allocator;
// expected-error@+1 {{expected one of the predefined allocators for the variables with the static storage: 'omp_default_mem_alloc', 'omp_large_cap_mem_alloc', 'omp_const_mem_alloc', 'omp_high_bw_mem_alloc', 'omp_low_lat_mem_alloc', 'omp_cgroup_mem_alloc', 'omp_pteam_mem_alloc' or 'omp_thread_mem_alloc'}}
#pragma omp allocate(d) allocator(allocator)
#pragma omp allocate(d) allocator(omp_thread_mem_alloc) // expected-note {{previous allocator is specified here}}
#pragma omp allocate(d) // expected-warning {{allocate directive specifies default allocator while previously used 'omp_thread_mem_alloc'}}

int c;
#pragma omp allocate(c) allocator(omp_thread_mem_alloc) // expected-note {{previous allocator is specified here}}
#pragma omp allocate(c) allocator(omp_high_bw_mem_alloc) // expected-warning {{allocate directive specifies 'omp_high_bw_mem_alloc' allocator while previously used 'omp_thread_mem_alloc'}}

