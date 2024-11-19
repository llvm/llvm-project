// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=52 %s

typedef enum omp_allocator_handle_t {
      omp_null_allocator = 0,
      omp_default_mem_alloc = 1,
      omp_large_cap_mem_alloc = 2,
      omp_const_mem_alloc = 3,
      omp_high_bw_mem_alloc = 4,
      omp_low_lat_mem_alloc = 5,
      omp_cgroup_mem_alloc = 6,
      omp_pteam_mem_alloc = 7,
      omp_thread_mem_alloc = 8,
} omp_allocator_handle_t;

int myAlloc() {
  return 100;
}

int main() {
  int a, b, c;
  // expected-error@+4 {{expected '('}}
  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(c) allocate(allocator
  // expected-error@+6 {{expected expression}}
  // expected-error@+5 {{expected ')'}}
  // expected-note@+4 {{to match this '('}}
  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(c) allocate(allocator(
  // expected-error@+4 {{expected expression}}
  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(c) allocate(allocator()
  // expected-error@+2 {{expected expression}}
  // expected-error@+1 {{expected expression}}
  #pragma omp scope private(c) allocate(allocator())
  // expected-error@+6 {{expected ')'}}
  // expected-note@+5 {{to match this '('}}
  // expected-error@+4 {{missing ':' after allocator modifier}}
  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(c) allocate(allocator(omp_default_mem_alloc
  // expected-error@+6 {{missing ':' after allocator modifier}}
  // expected-error@+5 {{expected expression}}
  // expected-error@+4 {{expected ')'}}
  // expected-note@+3 {{to match this '('}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(c) allocate(allocator(omp_large_cap_mem_alloc:
  // expected-error@+4 {{missing ':' after allocator modifier}}
  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(c) allocate(allocator(omp_const_mem_alloc)
  // expected-error@+2 {{missing ':' after allocator modifier}}
  // expected-error@+1 {{expected expression}}
  #pragma omp scope private(c) allocate(allocator(omp_high_bw_mem_alloc))
  // expected-error@+1 {{expected expression}}
  #pragma omp scope private(c) allocate(allocator(omp_low_lat_mem_alloc):)
  // expected-error@+6 {{expected ')'}}
  // expected-note@+5 {{to match this '('}}
  // expected-error@+4 {{missing ':' after allocator modifier}}
  // expected-error@+3 {{expected expression}}
  // expected-error@+2 {{expected ')'}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(c) allocate(allocator(omp_cgroup_mem_alloc:)
  // expected-error@+4 {{expected ')'}}
  // expected-note@+3 {{to match this '('}}
  // expected-error@+2 {{missing ':' after allocator modifier}}
  // expected-error@+1 {{expected expression}}
  #pragma omp scope private(c) allocate(allocator(omp_pteam_mem_alloc:))
  // expected-error@+4 {{expected ')'}}
  // expected-note@+3 {{to match this '('}}
  // expected-error@+2 {{missing ':' after allocator modifier}}
  // expected-error@+1 {{expected expression}}
  #pragma omp scope private(c) allocate(allocator(omp_thread_mem_alloc:c))
  // expected-error@+1 {{expected variable name}}
  #pragma omp scope private(c) allocate(allocator(omp_const_mem_alloc):1)
  // expected-error@+1 {{expected variable name}}
  #pragma omp scope private(c) allocate(allocator(omp_const_mem_alloc):-10)
  // expected-error@+4 {{expected ',' or ')' in 'allocate' clause}}
  // expected-error@+3 {{expected ')'}}
  // expected-warning@+2 {{extra tokens at the end of '#pragma omp scope' are ignored}}
  // expected-note@+1 {{to match this '('}}
  #pragma omp scope private(a,b,c) allocate(allocator(omp_const_mem_alloc):c:b;a)
  // expected-error@+1 {{initializing 'const omp_allocator_handle_t' with an expression of incompatible type 'int'}}
  #pragma omp scope private(c,a,b) allocate(allocator(myAlloc()):a,b,c)
  // expected-error@+2 {{missing ':' after allocator modifier}}
  // expected-error@+1 {{expected expression}}
  #pragma omp scope private(c) allocate(allocator(omp_default_mem_alloc);c)
  ++a;
}
