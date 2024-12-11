
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits -verify %t
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits %t > %t.cc_out 2> %t.cc_out
// RUN: FileCheck --check-prefix=DPF-CHECK %s --input-file=%t.cc_out

#include <ptrcheck.h>
#include <stddef.h>

// expected-note@+1 2{{passing argument to parameter here}}
size_t my_strlen(const char* __null_terminated);
int my_memcmp(const void*__sized_by(n) s1, const void *__sized_by(n) s2, size_t n);

typedef int proc_t;

const char* cs_identity_get(proc_t proc);

#define SIGNING_ID "signing_id"
#define SIGNING_ID_LEN (sizeof(SIGNING_ID) - 1)

void call(proc_t proc) {
    // expected-note@+2 2{{consider adding '__null_terminated' to 'signing_id'}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:17-[[@LINE+1]]:17}:"__null_terminated "
    const char *signing_id = NULL;
    // =========================================================================
    // error: assigning to 'const char *__bidi_indexable' from incompatible type
    // 'const char *__single __terminated_by(0)' (aka 'const char *__single')
    // requires a linear search for the terminator; use
    // '__terminated_by_to_indexable()' to perform this conversion explicitly
    //
    // **NOTE**: This is __null_terminated -> __bidi conversion
    //
    // ---
    // Multiple choice fix-its (each to be emitted on a separate note)
    // 1. Use builtin to perform conversion:
    // ```
    // (signing_id = __null_terminated_to_indexable(cs_identity_get(proc)))
    // ```
    // Note `__terminated_by_to_indexable` should be suggested instead if the thing we want to convert is a __terminated_by that's
    // not a __null_terminated.
    //
    // 2. Add `__null_terminated` to the declaration of the local.
    //
    // ```
    //  const char *__null_terminated signing_id = NULL;
    // ```
    //
    // Note: this breaks the call to `my_memcmp` which is why this a fix-it on a note (low confidence fix-it),
    // rather than the top-level diagnostic (high-confidence fix-it).
    // =========================================================================
    // expected-error@+7{{assigning to 'const char *__bidi_indexable' from incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
    // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:18-[[@LINE+5]]:18}:"__null_terminated_to_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:39-[[@LINE+4]]:39}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:18-[[@LINE+2]]:18}:"__unsafe_null_terminated_to_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:39-[[@LINE+1]]:39}:")"
    signing_id = cs_identity_get(proc);
    // =========================================================================
    // error: passing 'const char *__bidi_indexable' to parameter of
    // incompatible type 'const char *__single __terminated_by(0)'
    // (aka 'const char *__single') is an unsafe operation;
    // use '__unsafe_terminated_by_from_indexable()' or
    // '__unsafe_forge_terminated_by()' to perform this conversion
    //
    // **NOTE**: This is  __bidi conversion -> __null_terminated conversion
    // ---
    // Multiple choice fix-its (each to be emitted on a separate note)
    //
    // 1. Use O(N) search builtin
    //
    // my_strlen(__unsafe_null_terminated_from_indexable(signing_id))
    //
    // 2. Use O(1) search builtin
    //
    // my_strlen(__unsafe_null_terminated_from_indexable(signing_id, <# pointer to null terminator #>))
    //
    // 3. Add `__null_terminated` to the declaration of the local.
    //
    // ```
    //  const char *__null_terminated signing_id = NULL;
    // ```
    // =========================================================================
    // expected-error@+7{{passing 'const char *__bidi_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:20-[[@LINE+5]]:20}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:30-[[@LINE+4]]:30}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:20-[[@LINE+2]]:20}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:30-[[@LINE+1]]:30}:", <# pointer to null terminator #>)"
    if ((my_strlen(signing_id) == SIGNING_ID_LEN))
        return;
    if (my_memcmp(signing_id, SIGNING_ID, SIGNING_ID_LEN) == 0)
        return;
  // expected-error@+9{{initializing 'const char *__bidi_indexable' with an expression of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+8{{consider adding '__null_terminated' to 'var_init'}}
  // expected-note@+7{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+6]]:15-[[@LINE+6]]:15}:"__null_terminated "
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:26-[[@LINE+5]]:26}:"__null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:47-[[@LINE+4]]:47}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:26-[[@LINE+2]]:26}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:47-[[@LINE+1]]:47}:")"
  const char *var_init = cs_identity_get(proc);
}

const char *__bidi_indexable test_ret_null_to_bidi(proc_t proc) {
  // expected-error@+7{{returning 'const char *__single __terminated_by(0)' (aka 'const char *__single') from a function with incompatible result type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:31-[[@LINE+4]]:31}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:31-[[@LINE+1]]:31}:")"
  return cs_identity_get(proc);
}

__ptrcheck_abi_assume_bidi_indexable()
// expected-note@+2{{consider adding '__null_terminated' to 'const char *' return type of 'test_ret_null_to_imp_bidi'}}
// DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:13-[[@LINE+1]]:13}:"__null_terminated "
const char *test_ret_null_to_imp_bidi(proc_t proc) {
  // expected-error@+7{{returning 'const char *__single __terminated_by(0)' (aka 'const char *__single') from a function with incompatible result type 'const char *__bidi_indexable' requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:31-[[@LINE+4]]:31}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:31-[[@LINE+1]]:31}:")"
  return cs_identity_get(proc);
}

__ptrcheck_abi_assume_single()

const char *__null_terminated test_ret_bidi_to_null() {
  // expected-note@+2{{consider adding '__null_terminated' to 'local'}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:15-[[@LINE+1]]:15}:"__null_terminated "
  const char *local = "bidi_local";
  // expected-error@+7{{returning 'const char *__bidi_indexable' from a function with incompatible result type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:10-[[@LINE+5]]:10}:"__unsafe_null_terminated_from_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:15-[[@LINE+4]]:15}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:10-[[@LINE+2]]:10}:"__unsafe_null_terminated_from_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:15-[[@LINE+1]]:15}:", <# pointer to null terminator #>)"
  return local;
}

void test_cast_null_to_bidi(proc_t proc) {
  // expected-error@+9{{initializing 'const char *__bidi_indexable' with an expression of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+8{{consider adding '__null_terminated' to 'cast_var'}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+7]]:15-[[@LINE+7]]:15}:"__null_terminated "
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:26-[[@LINE+5]]:26}:"__null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:61-[[@LINE+4]]:61}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:26-[[@LINE+2]]:26}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:61-[[@LINE+1]]:61}:")"
  const char *cast_var = (const char *)cs_identity_get(proc);
}

void test_cast_bidi_to_null(const char *__bidi_indexable bidi_param) {
  // expected-error@+7{{initializing 'const char *__single __terminated_by(0)' (aka 'const char *__single') with an expression of incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:46-[[@LINE+5]]:46}:"__unsafe_null_terminated_from_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:70-[[@LINE+4]]:70}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:46-[[@LINE+2]]:46}:"__unsafe_null_terminated_from_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:70-[[@LINE+1]]:70}:", <# pointer to null terminator #>)"
  const char *__null_terminated null_local = (const char *)bidi_param;
}

// Constant array in struct variant

#define SIZE 4
struct Foo {
    // expected-note@+2{{consider adding '__null_terminated' to 'msg'}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:20-[[@LINE+1]]:20}:"__null_terminated "
    const char msg[SIZE];
    const char *msg_ptr;
};

void consumeFoo(struct Foo* f) {
    // =========================================================================
    // error:  passing 'const char[4]' to parameter of incompatible type
    // 'const char *__single __terminated_by(0)' (aka 'const char *__single')
    // is an unsafe operation; use '__unsafe_terminated_by_from_indexable()' or
    // '__unsafe_forge_terminated_by()' to perform this conversion
    //
    // **NOTE**: This is  __bidi conversion -> __null_terminated conversion
    // ---
    //
    // Multiple choice fix-its (each to be emitted on a separate note)
    //
    // 1. Use O(N) search builtin
    //
    // if (my_strlen(__unsafe_null_terminated_from_indexable(f->msg)))
    //
    // 2. Use O(1) search builtin
    //
    // if (my_strlen(__unsafe_null_terminated_from_indexable(f->msg, <# pointer to null terminator #>)))
    //
    // 3. Add `__null_terminated` to the declaration of the constant array
    //
    // ```
    // struct Foo {
    //   const char msg[__null_terminated SIZE];
    // };
    // ```
    // =========================================================================
    // expected-error@+7{{passing 'const char[4]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:19-[[@LINE+5]]:19}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:25-[[@LINE+4]]:25}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:19-[[@LINE+2]]:19}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:25-[[@LINE+1]]:25}:", <# pointer to null terminator #>)"
    if (my_strlen(f->msg)) {
        // do something
    }
    // expected-error@+9{{initializing 'const char *__bidi_indexable' with an expression of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
    // expected-note@+8{{consider adding '__null_terminated' to 'var_init'}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+7]]:17-[[@LINE+7]]:17}:"__null_terminated "
    // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:28-[[@LINE+5]]:28}:"__null_terminated_to_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:38-[[@LINE+4]]:38}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:28-[[@LINE+2]]:28}:"__unsafe_null_terminated_to_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:38-[[@LINE+1]]:38}:")"
    const char *var_init = f->msg_ptr;
    // expected-error@+7{{assigning to 'const char *__single __terminated_by(0)' (aka 'const char *__single') from incompatible type 'const char *__bidi_indexable' is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:18-[[@LINE+5]]:18}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:26-[[@LINE+4]]:26}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:18-[[@LINE+2]]:18}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:26-[[@LINE+1]]:26}:", <# pointer to null terminator #>)"
    f->msg_ptr = var_init;
}

void call1(proc_t proc) {
  const char *__bidi_indexable signing_id = NULL;
  // expected-error@+7{{assigning to 'const char *__bidi_indexable' from incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') requires a linear search for the terminator; use '__null_terminated_to_indexable()' to perform this conversion explicitly}}
  // expected-note@+6{{consider using '__null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound excludes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:16-[[@LINE+5]]:16}:"__null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:37-[[@LINE+4]]:37}:")"
  // expected-note@+3{{consider using '__unsafe_null_terminated_to_indexable()' to perform this conversion. Note this conversion requires a linear scan of memory to find the null terminator and the resulting upper bound includes the null terminator}}
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:16-[[@LINE+2]]:16}:"__unsafe_null_terminated_to_indexable("
  // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:37-[[@LINE+1]]:37}:")"
  signing_id = cs_identity_get(proc);
}

// expected-note@+1 {{passing argument to parameter here}}
void consume_null_param(const char * __null_terminated);
void test_implicit_bidi_to_null_param(void) {
    // expected-note@+2{{consider adding '__null_terminated' to 'local'}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:17-[[@LINE+1]]:17}:"__null_terminated "
    const char* local;
    // expected-error@+7{{passing 'const char *__bidi_indexable' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:24-[[@LINE+5]]:24}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:31-[[@LINE+4]]:31}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:24-[[@LINE+2]]:24}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:31-[[@LINE+1]]:31}:", <# pointer to null terminator #>)"
    consume_null_param((local));
}

// expected-note@+1 2{{passing argument to parameter here}}
void consume_long_null_param(long long int * __null_terminated);
void test_bitcast_bidi_to_null(void) {
    // expected-note@+2 2{{consider adding '__null_terminated' to 'local'}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:10-[[@LINE+1]]:10}:"__null_terminated "
    int* local;
    // expected-error@+7{{passing 'long long *__bidi_indexable' to parameter of incompatible type 'long long *__single __terminated_by(0)' (aka 'long long *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:29-[[@LINE+5]]:29}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:51-[[@LINE+4]]:51}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:29-[[@LINE+2]]:29}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:51-[[@LINE+1]]:51}:", <# pointer to null terminator #>)"
    consume_long_null_param((long long int*) local);  // Fixit doesn't fix anything: rdar://122840377
    // expected-error@+7{{passing 'long long *__bidi_indexable' to parameter of incompatible type 'long long *__single __terminated_by(0)' (aka 'long long *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
    // expected-note@+6{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+5]]:29-[[@LINE+5]]:29}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+4]]:67-[[@LINE+4]]:67}:")"
    // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+2]]:29-[[@LINE+2]]:29}:"__unsafe_null_terminated_from_indexable("
    // DPF-CHECK: fix-it:"{{.+}}null_terminated_bidi_indexable_conv.c.tmp":{[[@LINE+1]]:67-[[@LINE+1]]:67}:", <# pointer to null terminator #>)"
    consume_long_null_param((long long int*)(long long int*) local);
}
