// RUN: %clang_cc1 -verify -fopenmp %s -Wuninitialized

// RUN: %clang_cc1 -verify -fopenmp-simd %s -Wuninitialized

// RUN: %clang_cc1 -DOMP52 -verify=omp52 -fopenmp -fopenmp-version=52 %s -Wuninitialized

// RUN: %clang_cc1 -DOMP52 -verify=omp52 -fopenmp-simd  -fopenmp-version=52 %s -Wuninitialized

struct omp_alloctrait_t {};

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

int main(int argc, char **argv) {
  omp_alloctrait_t my_traits[10];
  omp_alloctrait_t *ptraits;
  omp_allocator_handle_t my_alloc = nullptr;
  const omp_allocator_handle_t c_my_alloc = my_alloc;
  omp_alloctrait_t aligned_traits[10];
  omp_allocator_handle_t aligned_alloc;

#ifndef OMP52

#pragma omp target uses_allocators // expected-error {{expected '(' after 'uses_allocator'}}
{}
#pragma omp target uses_allocators( // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected unqualified-id}}
{}
#pragma omp target uses_allocators() // expected-error {{expected unqualified-id}}
{}
#pragma omp target uses_allocators(omp_default_mem_alloc // expected-error {{expected ',' or ')' in 'uses_allocators' clause}} expected-error {{expected ')'}} expected-note {{to match this '('}}
{}
#pragma omp target uses_allocators(argc, // expected-error {{expected ')'}} expected-error {{expected variable of the 'omp_allocator_handle_t' type, not 'int'}} expected-note {{to match this '('}}
{}
#pragma omp target uses_allocators(argc > 0 ? omp_default_mem_alloc : omp_thread_mem_alloc) // expected-error {{expected ',' or ')' in 'uses_allocators' clause}} expected-error {{expected unqualified-id}} expected-error {{expected variable of the 'omp_allocator_handle_t' type, not 'int'}}
{}
#pragma omp target uses_allocators(omp_default_mem_alloc, omp_large_cap_mem_alloc, omp_const_mem_alloc, omp_high_bw_mem_alloc, omp_low_lat_mem_alloc, omp_cgroup_mem_alloc, omp_pteam_mem_alloc, omp_thread_mem_alloc)
{}
#pragma omp target uses_allocators(omp_default_mem_alloc(my_traits), omp_large_cap_mem_alloc(my_traits), omp_const_mem_alloc(my_traits), omp_high_bw_mem_alloc(my_traits), omp_low_lat_mem_alloc(my_traits), omp_cgroup_mem_alloc(my_traits), omp_pteam_mem_alloc(my_traits), omp_thread_mem_alloc(my_traits)) // expected-error 8 {{predefined allocator cannot have traits specified}} expected-note-re 8 {{predefined trait '{{omp_default_mem_alloc|omp_large_cap_mem_alloc|omp_const_mem_alloc|omp_high_bw_mem_alloc|omp_low_lat_mem_alloc|omp_cgroup_mem_alloc|omp_pteam_mem_alloc|omp_thread_mem_alloc}}' used here}}
{}
#pragma omp target uses_allocators(my_alloc, c_my_alloc) // expected-error {{non-predefined allocator must have traits specified}} expected-error {{expected variable of the 'omp_allocator_handle_t' type, not 'const omp_allocator_handle_t' (aka 'void **const')}}
{}
#pragma omp target uses_allocators(my_alloc() // expected-error {{expected unqualified-id}} expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{non-predefined allocator must have traits specified}}
{}
#pragma omp target uses_allocators(my_alloc()) // expected-error {{expected unqualified-id}} expected-error {{non-predefined allocator must have traits specified}}
{}
#pragma omp target uses_allocators(my_alloc(argc > 0 ? argv[0] : argv{1})) // expected-error {{expected ')'}} expected-note {{to match this '('}} expected-error {{expected constant sized array of 'omp_alloctrait_t' elements, not 'int'}}
{}
#pragma omp target uses_allocators(my_alloc(ptraits)) // expected-error {{expected constant sized array of 'omp_alloctrait_t' elements, not 'omp_alloctrait_t *'}}
{}
#pragma omp target uses_allocators(my_alloc(my_traits)) private(my_alloc) // expected-error {{allocators used in 'uses_allocators' clause cannot appear in other data-sharing or data-mapping attribute clauses}} expected-note {{defined as private}}
{}
#pragma omp target map(my_alloc, my_traits) uses_allocators(my_alloc(my_traits)) // expected-error {{allocators used in 'uses_allocators' clause cannot appear in other data-sharing or data-mapping attribute clauses}} expected-note {{used here}}
{}
#else
#pragma omp target teams uses_allocators(traits(my_traits): my_alloc; traits(aligned_traits): aligned_alloc)
{}
#pragma omp target teams uses_allocators(my_alloc)
{}
#pragma omp target teams uses_allocators(traits(my_traits): my_alloc, traits(aligned_traits): aligned_alloc) // omp52-error {{',' not allowed as separator in 'uses_allocators' clause, use ';' instead}}
{}
#pragma omp target teams uses_allocators(my_alloc(my_traits), aligned_alloc(aligned_traits)) // omp52-error {{old syntax 'allocator(expr)' on 'uses_allocators' clause was deprecated, use new syntax 'traits(expr): alloc'}} //omp52-error {{old syntax 'allocator(expr)' on 'uses_allocators' clause was deprecated, use new syntax 'traits(expr): alloc'}}
{}
#endif
  return 0;
}

