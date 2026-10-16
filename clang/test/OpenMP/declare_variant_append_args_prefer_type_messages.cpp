// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=60 -std=c++11 -o - %s

typedef void *omp_interop_t;

void foo_v1(float *A, float *B, omp_interop_t IOp);

// expected-error@+2 {{prefer_list item must be a string literal or constant integral expression}}
#pragma omp declare variant(foo_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr(1.0)}), target))
void foo_fr_float(float *A, float *B) {}

void bar_v1(float *A, omp_interop_t IOp);

// expected-error@+2 {{attr() argument must be a string literal}}
#pragma omp declare variant(bar_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr(1)}), target))
void bar_attr_int(float *A) {}

void baz_v1(float *A, omp_interop_t IOp);

// expected-error@+2 {{attr() argument 'cuda_prop' must start with the 'ompx_' prefix}}
#pragma omp declare variant(baz_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("cuda_prop")}), target))
void baz_attr_no_prefix(float *A) {}

void qux_v1(float *A, omp_interop_t IOp);

// expected-error@+2 {{attr() argument 'ompx_a,b' must not contain a comma}}
#pragma omp declare variant(qux_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("ompx_a,b")}), target))
void qux_attr_comma(float *A) {}

// Edge cases for attr() and fr().
void edge_v1(float *A, omp_interop_t IOp);

// expected-error@+2 {{attr() argument 'ompx_a,b,c' must not contain a comma}}
#pragma omp declare variant(edge_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("ompx_a,b,c")}), target))
void edge_attr_multi_commas(float *A) {}

void edge_v2(float *A, omp_interop_t IOp);

// expected-error@+2 {{attr() argument '' must start with the 'ompx_' prefix}}
#pragma omp declare variant(edge_v2) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("")}), target))
void edge_attr_empty(float *A) {}

// Valid cases -- no diagnostics expected.
void valid_v1(float *A, omp_interop_t IOp);

#pragma omp declare variant(valid_v1) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr("cuda")}), target))
void valid_fr_string(float *A) {}

void valid_v2(float *A, omp_interop_t IOp);

#pragma omp declare variant(valid_v2) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr(1)}), target))
void valid_fr_int(float *A) {}

void valid_v3(float *A, omp_interop_t IOp);

#pragma omp declare variant(valid_v3) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("ompx_myattr")}), target))
void valid_attr(float *A) {}

void valid_v4(float *A, omp_interop_t IOp);

#pragma omp declare variant(valid_v4) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("ompx_prop")}), target))
void valid_attr_only(float *A) {}

void valid_v5(float *A, omp_interop_t IOp);

#pragma omp declare variant(valid_v5) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr(1), attr("ompx_prop")}), target))
void valid_combined(float *A) {}

void valid_v6(float *A, omp_interop_t IOp);

#pragma omp declare variant(valid_v6) match(construct={dispatch}) \
  append_args(interop(prefer_type({attr("ompx_")}), target))
void valid_attr_prefix_only(float *A) {}

void valid_v7(float *A, omp_interop_t IOp);

#pragma omp declare variant(valid_v7) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr("")}), target))
void valid_fr_empty_string(float *A) {}

// Template case: fr() argument becomes invalid at instantiation.
template <typename T>
void tmpl_v1(T *A, omp_interop_t IOp);

// expected-error@+2 {{prefer_list item must be a string literal or constant integral expression}}
#pragma omp declare variant(tmpl_v1<int>) match(construct={dispatch}) \
  append_args(interop(prefer_type({fr(1.5)}), target))
void tmpl_fr_invalid(int *A) {}
