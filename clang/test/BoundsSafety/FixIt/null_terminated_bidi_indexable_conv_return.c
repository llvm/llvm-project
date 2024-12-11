
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits -verify %t
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits %t > %t.cc_out 2> %t.cc_out
// RUN: FileCheck --check-prefix=DPF-CHECK %s --input-file=%t.cc_out

#include <ptrcheck.h>

typedef int proc_t;

const char *cs_identity_get(proc_t proc);

const char *test_ret_null_to_imp_bidi_ret_var(proc_t proc) {
  // expected-error@+9{{initializing 'const char *__bidi_indexable' with an expression of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+8 2{{consider adding '__null_terminated' to 'ret'}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+7]]:15-[[@LINE+7]]:15}:"__null_terminated "
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:21-[[@LINE+5]]:21}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:42-[[@LINE+4]]:42}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:21-[[@LINE+2]]:21}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:42-[[@LINE+1]]:42}:")"
  const char *ret = cs_identity_get(proc);
  // expected-error@+7{{returning 'const char *__bidi_indexable' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__unsafe_null_terminated_from_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:13-[[@LINE+4]]:13}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_from_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:", <# pointer to null terminator #>)"
  return ret;
}

const char *test_ret_null_to_imp_bidi_ret_var_attributed(proc_t proc) {
  const char *__null_terminated ret = cs_identity_get(proc);
  return ret;
}

const char *__bidi_indexable test_ret_null_to_imp_bidi_ret_var_attributed_explicit(proc_t proc) {
  // expected-error@+9{{initializing 'const char *__bidi_indexable' with an expression of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+8{{consider adding '__null_terminated' to 'ret'}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+7]]:15-[[@LINE+7]]:15}:"__null_terminated "
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:21-[[@LINE+5]]:21}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:42-[[@LINE+4]]:42}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:21-[[@LINE+2]]:21}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:42-[[@LINE+1]]:42}:")"
  const char *ret = cs_identity_get(proc);
  return ret;
}

__ptrcheck_abi_assume_bidi_indexable()

// The DPF-CHECK-DAGs on `test_ret_null_to_bidi` test that all prototypes prior to its definition get fixits.
// DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:"__null_terminated "
const char *test_ret_null_to_bidi(proc_t proc);

// DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:"__null_terminated "
const char *test_ret_null_to_bidi(proc_t proc);

// expected-note@+2{{consider adding '__null_terminated' to 'const char *' return type of 'test_ret_null_to_bidi'}}
// DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:"__null_terminated "
const char *test_ret_null_to_bidi(proc_t proc) {
  // expected-error@+7{{returning 'const char *__single __terminated_by(0)' (aka 'const char *__single') from a function with incompatible result type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:31-[[@LINE+4]]:31}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:31-[[@LINE+1]]:31}:")"
  return cs_identity_get(proc);
}

const char *test_assign_null_to_imp_bidi(const char *__null_terminated foo) {
  const char *bar[42];
  // expected-error@+7{{assigning to 'const char *__bidi_indexable' from incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:12-[[@LINE+5]]:12}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:15-[[@LINE+4]]:15}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:12-[[@LINE+2]]:12}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:15-[[@LINE+1]]:15}:")"
  bar[1] = foo;
  return bar[1];
}

const char *test_assign_null_to_imp_bidi_annotate(const char *__null_terminated foo) {
  const char *bar[42];
  bar[1] = __null_terminated_to_indexable(foo);
  return bar[1];
}

// expected-note@+2{{consider adding '__null_terminated' to 'const char *' return type of 'test_ret_null_to_imp_bidi'}}
// DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:"__null_terminated "
const char *test_ret_null_to_imp_bidi(const char *__null_terminated foo) {
  // expected-error@+7{{returning 'const char *__single __terminated_by(0)' (aka 'const char *__single') from a function with incompatible result type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:13-[[@LINE+4]]:13}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:")"
  return foo;
}

// FIXME: rdar://125936876
const char *test_ret_null_to_bidi(proc_t proc);

// expected-note@+2{{consider adding '__null_terminated' to 'const char *' return type of 'test_ret_null_to_imp_bidi_param'}}
// DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:"__null_terminated "
const char *test_ret_null_to_imp_bidi_param(const char *__null_terminated foo, const char *param) {
  // expected-error@+7{{returning 'const char *__single __terminated_by(0)' (aka 'const char *__single') from a function with incompatible result type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:13-[[@LINE+4]]:13}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:")"
  return foo;
}

#define CC_TY const char*

CC_TY __null_terminated return_macro(void);

// expected-note@+2{{consider adding '__null_terminated' to 'const char *' return type of 'test_return_macro'}}
// DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:7-[[@LINE+1]]:7}:"__null_terminated "
CC_TY test_return_macro(void) {
  const char *__null_terminated a = return_macro();
  // expected-error@+7{{returning 'const char *__single __terminated_by(0)' (aka 'const char *__single') from a function with incompatible result type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:11-[[@LINE+4]]:11}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:11-[[@LINE+1]]:11}:")"
  return a;
}

const char *test_ret_null_to_imp_bidi_no_fix(const char *__null_terminated foo) {
  const char *bar[42];
  // expected-error@+7{{assigning to 'const char *__bidi_indexable' from incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+5]]:12-[[@LINE+5]]:12}:"__null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+4]]:15-[[@LINE+4]]:15}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+2]]:12-[[@LINE+2]]:12}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK-DAG: fix-it:"{{.+}}null_terminated_bidi_indexable_conv_return.c.tmp":{[[@LINE+1]]:15-[[@LINE+1]]:15}:")"
  bar[1] = foo;
  return 0;
}

int *test_imp_single_ret_imp_bidi();

int *test_imp_single_ret_imp_bidi() {
  int *__single local;
  return local;
}

__ptrcheck_abi_assume_single()

int *test_imp_single_ret() {
  int *local;
  return local;
}

int *__bidi_indexable test_bidi_ret() {
  // expected-note@+1{{pointer 'local' declared here}}
  int *__single local;
  // expected-warning@+1{{returning type 'int *__single' from a function with result type 'int *__bidi_indexable' results in a __bidi_indexable pointer that will trap if a non-zero offset is dereferenced. consider adding '__counted_by' to 'local'}}
  return local;
}

int *test_imp_single_ret_bidi() {
  int *__bidi_indexable local;
  return local;
}