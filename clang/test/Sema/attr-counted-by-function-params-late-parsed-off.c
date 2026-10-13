// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify %s
// RUN: %clang_cc1 -DNEEDS_LATE_PARSING -fsyntax-only -verify %s

// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fno-experimental-late-parse-attributes -fsyntax-only -verify=ok %s
// RUN: %clang_cc1 -UNEEDS_LATE_PARSING -fsyntax-only -verify=ok %s

// Without late parsing, a parameter's count is parsed where it is written, so
// it can only name a parameter declared before it.

#define __counted_by(f)  __attribute__((counted_by(f)))
#define __sized_by_or_null(f)  __attribute__((sized_by_or_null(f)))

typedef int *int_ptr;

#ifdef NEEDS_LATE_PARSING

// expected-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref(int *__counted_by(count) buf, int count);
// expected-error@+1{{use of undeclared identifier 'count'}}
void fwd_ref_sized_or_null(void *__sized_by_or_null(count) buf, int count);
// expected-error@+1{{use of undeclared identifier 'len'}}
void inner_fwd_ref(void (*cb)(int *__counted_by(len) p, int len));
// expected-error@+1{{use of undeclared identifier 'count'}}
void trailing_fwd_ref(int *buf __counted_by(count), int count);
// expected-error@+1{{use of undeclared identifier 'count'}}
void declspec_fwd_ref(int_ptr __counted_by(count) buf, int count);
// expected-error@+1{{use of undeclared identifier 'count'}}
void array_fwd_ref(int arr[] __counted_by(count), int count);
// expected-error@+1{{use of undeclared identifier 'len'}}
void out_fwd_ref(int *__counted_by(*len) *buf, int *len);
// A callback's return count names parameters declared after it.
// expected-error@+1{{use of undeclared identifier 'len'}}
void cb_return(int *__counted_by(len) (*cb)(int len));

#else

// ok-no-diagnostics
void back_ref(int count, int *__counted_by(count) buf);
void back_ref_sized_or_null(int count, void *__sized_by_or_null(count) buf);
void inner_back_ref(void (*cb)(int len, int *__counted_by(len) p));
void trailing_back_ref(int count, int *buf __counted_by(count));
void declspec_back_ref(int count, int_ptr __counted_by(count) buf);
void array_back_ref(int count, int arr[] __counted_by(count));
void out_back_ref(int *len, int *__counted_by(*len) *buf);

#endif
