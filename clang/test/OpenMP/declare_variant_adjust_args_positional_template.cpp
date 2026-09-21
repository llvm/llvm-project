// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -fsyntax-only -verify %s

// OpenMP 6.0 [5.2.1]: a dependent parameter-list item is accepted at the
// template definition and rechecked only once substitution makes it
// non-dependent (S6). SemaTemplateInstantiateDecl re-invokes
// ActOnOpenMPDeclareVariantDirective per specialization, so nothing at
// definition time is lost by deferring.

template <int N>
void v_tmpl(int *aaa, int *bbb, ...);

// No error here: 'N' is value-dependent at definition time.
#pragma omp declare variant(v_tmpl<N>) match(construct={dispatch}) \
  adjust_args(need_device_ptr: N)
template <int N>
void tmpl_pos(int *aaa, int *bbb, ...) {}

// Clean once substituted: position 2 is 'bbb', a valid, positive position.
template void tmpl_pos<2>(int *, int *, ...);

// The positive property (OpenMP 6.0 [5.2.1] p162 L30-31) is only checkable
// once 'N' is substituted.
// expected-error@15 {{argument to 'adjust_args' clause must be a strictly positive integer value}}
// expected-note@+1 {{in instantiation of function template specialization 'tmpl_pos<0>' requested here}}
template void tmpl_pos<0>(int *, int *, ...);

template <int N>
void v_tmpl_range(int *aaa, int *bbb, int *ccc, ...);

// A dependent range bound is accepted at definition time too.
#pragma omp declare variant(v_tmpl_range<N>) match(construct={dispatch}) \
  adjust_args(need_device_ptr: N:N + 1)
template <int N>
void tmpl_range(int *aaa, int *bbb, int *ccc, ...) {}

// Clean once substituted: range 2:3 covers 'bbb' and 'ccc'.
template void tmpl_range<2>(int *, int *, int *, ...);

// The positive property on a range bound is likewise only checkable once 'N'
// is substituted.
// expected-error@33 {{argument to 'adjust_args' clause must be a strictly positive integer value}}
// expected-note@+1 {{in instantiation of function template specialization 'tmpl_range<0>' requested here}}
template void tmpl_range<0>(int *, int *, int *, ...);

template <int N>
void v_tmpl_offset(int *aaa, int *bbb, ...);

// A dependent 'omp_num_args' logical offset is accepted at definition time.
#pragma omp declare variant(v_tmpl_offset<N>) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args-N:omp_num_args)
template <int N>
void tmpl_offset(int *aaa, int *bbb, ...) {}

// Clean once substituted: offset 1 is non-negative.
template void tmpl_offset<1>(int *, int *, ...);

// The non-negative property on the offset (OpenMP 6.0 [5.2.1] p163 L1) is
// likewise only checkable once 'N' is substituted.
// expected-error@51 {{argument to 'adjust_args' clause must be a non-negative integer value}}
// expected-note@+1 {{in instantiation of function template specialization 'tmpl_offset<-1>' requested here}}
template void tmpl_offset<-1>(int *, int *, ...);
