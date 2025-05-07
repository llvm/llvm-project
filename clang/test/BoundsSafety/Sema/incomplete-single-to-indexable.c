
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -flax-vector-conversions=none -ffreestanding -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -mvscale-min=4 -mvscale-max=4 -flax-vector-conversions=none -ffreestanding -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>
#include <arm_sve.h>
struct oq_struct;

typedef struct oq_struct* oq_struct_t;
#define NULL ((void *__single)0)

oq_struct_t __unsafe_indexable foo();

oq_struct_t test() {
  oq_struct_t local = 0; // expected-note{{pointer 'local' declared here}}
  // expected-error@+1{{cannot assign to indexable pointer with type 'struct oq_struct *__bidi_indexable' from __single pointer to incomplete type 'oq_struct_t __single' (aka 'struct oq_struct *__single'); consider declaring pointer 'local' as '__single'}}
  local = __unsafe_forge_single(oq_struct_t, foo());
  return local;
}

void test_sve(svint8_t *arg) {
  // expected-error@+1{{cannot initialize indexable pointer with type 'svint8_t *__bidi_indexable' (aka '__SVInt8_t *__bidi_indexable') from __single pointer to incomplete type 'svint8_t *__single' (aka '__SVInt8_t *__single'); consider declaring pointer 'local' as '__single'}}
  svint8_t *local = arg; // expected-note{{pointer 'local' declared here}}
}

typedef void(func_t)(void);
// function type is okay because it will always be '__single'
void test_func(func_t *arg) {
  func_t *local = arg;
}

void test_null() {
  int *impl_bidi_ptr = NULL;
  int *__bidi_indexable bidi_ptr = NULL;
  oq_struct_t __indexable ind_ptr = NULL;
}

void test_null_to_bidi() {
  int *impl_bidi_ptr = (int *__bidi_indexable)NULL;
  int *impl_bidi_ptr2 = (int *__indexable)NULL;
  int *impl_bidi_ptr3 = (void *)(char *)NULL;
}

// The extra note is omitted because another note about the param is already emitted
// and there's no FixIt.
// expected-note@+1{{passing argument to parameter 'param' here}}
void bidi_sink(void* __bidi_indexable param);

// The extra note is omitted because another note about the param is already emitted
// and there's no FixIt.
// expected-note@+1{{passing argument to parameter here}}
void bidi_sink_no_param_name(void* __bidi_indexable);

int* __bidi_indexable single_opaque_assigned_to_indexable_ret(
  void* __single p) {
    // expected-error@+1{{cannot return __single pointer to incomplete type 'void *__single' when return type is an indexable pointer 'int *__bidi_indexable'; consider making the return type '__single'}}
    return p;
}

void single_opaque_arg_passed(void* __single p) {
    // expected-error@+1{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the 'param' parameter '__single'}}
    bidi_sink(p);
}

void single_opaque_arg_passed2(void* __single p) {
    // expected-error@+1{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the parameter '__single'}}
    bidi_sink_no_param_name(p);
}

__ptrcheck_abi_assume_bidi_indexable();

// expected-note@+1{{passing argument to parameter 'implicit_bidi' here}}
void implicit_bidi_sink(void* implicit_bidi);

// expected-note@+3{{pointer 'implicit_bidi2' declared here}} // This is emitted due to a FixIt needs to be attached to a diagnostic and we can't attach it to the error.
// expected-note@+2{{passing argument to parameter 'implicit_bidi2' here}}
// expected-note@+1{{passing argument to parameter 'implicit_bidi2' here}}
void implicit_bidi_sink2(void* implicit_bidi2);

__ptrcheck_abi_assume_single();

void single_opaque_arg_passed_to_implicit_bidi_param(void* __single p) {
    // expected-error@+1{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the 'implicit_bidi' parameter '__single'}}
    implicit_bidi_sink(p);
}

void single_opaque_arg_passed_to_implicit_bidi_param2(void* __single p) {
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the 'implicit_bidi2' parameter '__single'}}
    // expected-error@+2{{cannot pass __single pointer to incomplete type argument 'void *__single' when parameter is an indexable pointer 'void *__bidi_indexable'; consider making the 'implicit_bidi2' parameter '__single'}}
    implicit_bidi_sink2(p);
    implicit_bidi_sink2(p);
}
