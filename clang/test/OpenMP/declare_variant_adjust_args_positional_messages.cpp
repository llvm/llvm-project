// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=60 \
// RUN:   -x c++ -std=c++14 -ferror-limit 100 -fsyntax-only -verify=expected %s

// The same syntax must still be rejected before OpenMP 6.0, which pins that the
// new parsing is version-gated.
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fopenmp -fopenmp-version=51 \
// RUN:   -x c++ -std=c++14 -ferror-limit 100 -DOMP51 -fsyntax-only \
// RUN:   -verify=omp51 %s

int G; // expected-note 2 {{declared here}}

void v1(int *aaa, int *bbb, ...);

#ifndef OMP51

// OpenMP 6.0 [5.2.1] p162: 'omp_num_args' may only be followed by a signed
// constant logical offset, so no other operator may continue the bound.
// expected-error@+2 {{'omp_num_args' may only be followed by '+' or '-' and a constant logical offset}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args/2)
void f1(int *aaa, int *bbb, ...);

// expected-error@+2 {{'omp_num_args' may only be followed by '+' or '-' and a constant logical offset}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args*2)
void f2(int *aaa, int *bbb, ...);

// 'omp_num_args' is recognised by spelling, and only where a bound starts. It
// has no declaration, so anywhere else it is just an unknown identifier.
// expected-error@+2 {{use of undeclared identifier 'omp_num_args'}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 2*omp_num_args)
void f3(int *aaa, int *bbb, ...);

// expected-error@+2 {{use of undeclared identifier 'omp_num_args'}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: (omp_num_args):2)
void f4(int *aaa, int *bbb, ...);

// A sign must be followed by an actual offset expression.
// expected-error@+2 {{expected expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args +)
void f5(int *aaa, int *bbb, ...);

// expected-error@+2 {{expected expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:omp_num_args-)
void f6(int *aaa, int *bbb, ...);

// The logical offset must be a constant integer expression.
// expected-error@+3 {{expression is not an integral constant expression}}
// expected-note@+2 {{read of non-const variable 'G' is not allowed in a constant expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args-G:omp_num_args)
void f7(int *aaa, int *bbb, ...);

// expected-error@+2 {{integral constant expression must have integral or unscoped enumeration type, not 'double'}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args-1.5:omp_num_args)
void f8(int *aaa, int *bbb, ...);

// OpenMP 6.0 [5.2.1] p162 lists three forms of parameter list item, and a bare
// 'omp_num_args' is not one of them: it is legal only as a range bound.
// expected-error@+2 {{'omp_num_args' is only allowed as a bound of a parameter range}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args)
void f9(int *aaa, int *bbb, ...);

// expected-error@+2 {{'omp_num_args' is only allowed as a bound of a parameter range}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args-1)
void f10(int *aaa, int *bbb, ...);

// The rejection is per item, not only for the first one.
// expected-error@+2 {{'omp_num_args' is only allowed as a bound of a parameter range}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: aaa, omp_num_args)
void f11(int *aaa, int *bbb, ...);

// A range has exactly one colon, so a third bound is a malformed item rather
// than a silently mis-parsed one.
// expected-error@+2 {{expected ',' or ')' in 'adjust_args' clause}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:2:3)
void f12(int *aaa, int *bbb, ...);

// A non-constant, non-parameter item is still not a position: it must be a
// constant integer expression (OpenMP 6.0 [5.2.1] p162 L30-31).
// expected-error@+3 {{expression is not an integral constant expression}}
// expected-note@+2 {{read of non-const variable 'G' is not allowed in a constant expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: G)
void f13(int *aaa, int *bbb, ...);

// The list must have at least one item.
// expected-error@+2 {{expected expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: )
void f14(int *aaa, int *bbb, ...);

// A trailing separator starts an item that is not there.
// expected-error@+2 {{expected expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:2,)
void f15(int *aaa, int *bbb, ...);

// A leading separator is not an omitted item either; only a colon may open one.
// expected-error@+2 {{expected expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: , 1)
void f16(int *aaa, int *bbb, ...);

// The list loop must give up at the end of the directive rather than spin on a
// clause that is never closed.
// expected-error@+3 {{expected ')'}}
// expected-note@+2 {{to match this '('}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:2
void f17(int *aaa, int *bbb, ...);

// A malformed item is skipped up to the separator, so it neither swallows nor
// duplicates a diagnostic for the item that follows it.
// expected-error@+2 {{'omp_num_args' is only allowed as a bound of a parameter range}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args, 2)
void f18(int *aaa, int *bbb, ...);

// OpenMP 6.0 [5.2.1] p162 L30-31: a position has the positive property.
// expected-error@+2 {{argument to 'adjust_args' clause must be a strictly positive integer value}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 0)
void h1(int *aaa, int *bbb, ...);

// expected-error@+2 {{argument to 'adjust_args' clause must be a strictly positive integer value}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: -1)
void h2(int *aaa, int *bbb, ...);

// A plain range bound has the positive property too.
// expected-error@+2 {{argument to 'adjust_args' clause must be a strictly positive integer value}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 0:5)
void h3(int *aaa, int *bbb, ...);

// The logical offset has the non-negative property, checked independently of
// its already-verified constant property (OpenMP 6.0 [5.2.1] p163 L1).
// expected-error@+2 {{argument to 'adjust_args' clause must be a non-negative integer value}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args-(-1):omp_num_args)
void h4(int *aaa, int *bbb, ...);

// The duplicate restriction (OpenMP 6.0 [5.2.1] p162) applies to positions,
// not only to names.
// expected-error@+2 {{'adjust_arg' argument 2 used in multiple clauses}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 2, 2)
void h5(int *aaa, int *bbb, ...);

// A name and a position that happen to resolve to the same parameter are two
// distinct items (OpenMP 6.0 [5.2.1] p162), so this is accepted, not a
// duplicate.
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: aaa, 1)
void h6(int *aaa, int *bbb, ...);

// Not a name, a range, or a position: none of the three forms is satisfied.
// expected-error@+2 {{expected a parameter name, a parameter position, or a parameter range in 'adjust_args' clause}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1.5)
void h7(int *aaa, int *bbb, ...);

// need_device_addr's reference-type restriction is not scoped to named items
// (OpenMP 6.0 [9.6.2] p332 L31-33), so a position is checked too.
// expected-error@+2 {{expected reference type argument on 'adjust_args' clause with 'need_device_addr' modifier}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_addr: 1)
void h8(int *aaa, int *bbb, ...);

// A huge literal upper bound must not turn range resolution into an unbounded
// loop: out-of-range positions are dropped (OpenMP 6.0 [9.6.2] p332 L1-2), not
// enumerated one at a time up to the written value.
void v2(int &aaa, int &bbb, ...);
#pragma omp declare variant(v2) match(construct={dispatch}) \
  adjust_args(need_device_addr: 1:9223372036854775807)
void h9(int &aaa, int &bbb, ...);

// need_device_addr's reference-type check via a range covers every position
// it sweeps, but reports only one diagnostic per written item, not one per
// position (OpenMP 6.0 [9.6.2] p332 L31-33).
// expected-error@+2 {{expected reference type argument on 'adjust_args' clause with 'need_device_addr' modifier}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_addr: 1:2)
void h10(int *aaa, int *bbb, ...);

// 'lb > ub' specifies no parameters and is accepted silently: the spec places
// no restriction on it (OpenMP 6.0 [5.2.1] p163).
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 5:2)
void h11(int *aaa, int *bbb, ...);

#else // OMP51

// Before 6.0 a range is not parsed at all: the list stops at the colon, so the
// clause is left unterminated and the leading bound is then rejected by Sema.
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 1:2) // omp51-error {{expected ',' or ')' in 'adjust_args' clause}} omp51-error {{expected ')'}} omp51-note {{to match this '('}} omp51-error {{expected reference to one of the parameters of function 'g1'}}
void g1(int *aaa, int *bbb, ...);

// A position is not a parameter name before 6.0.
// omp51-error@+2 {{expected reference to one of the parameters of function 'g2'}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: 2)
void g2(int *aaa, int *bbb, ...);

// An omitted lower bound is not an item, so the list is simply empty.
// omp51-error@+2 {{expected expression}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: :)
void g3(int *aaa, int *bbb, ...);

// 'omp_num_args' is not recognised at all, and is looked up as a name.
// omp51-error@+2 {{use of undeclared identifier 'omp_num_args'}}
#pragma omp declare variant(v1) match(construct={dispatch}) \
  adjust_args(need_device_ptr: omp_num_args-1:omp_num_args)
void g4(int *aaa, int *bbb, ...);

#endif // OMP51
