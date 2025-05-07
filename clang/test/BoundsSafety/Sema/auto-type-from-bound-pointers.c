

// RUN: %clang_cc1 -fbounds-safety -verify %s
// RUN: %clang_cc1 -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

struct SequencePtrs {
  char *__ended_by(iter) start;
  char *__ended_by(end) iter;
  char *end;
};

void TestEndedBy(struct SequencePtrs *sp) {
  __auto_type local_start = sp->start; // expected-error{{passing '__ended_by' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_iter = sp->iter; // expected-error{{passing '__ended_by' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_end = sp->end; // expected-error{{passing end pointer as __auto_type initializer is not yet supported}}
}

void TestCountedBy(int *__counted_by(len) ptr, int len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{passing '__counted_by' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_len = len;
  __auto_type local_counted_ptr_addrof = &ptr; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len_addrof = &len;         // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
}

void TestSizedBy(void *__sized_by(len) ptr, unsigned len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{passing '__sized_by' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_len = len;
  __auto_type local_counted_ptr_addrof = &ptr; // expected-error{{pointer with '__sized_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len_addrof = &len;         // expected-error{{variable referred to by '__sized_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
}

void TestOutCountedBy(int *__counted_by(*len) *ptr, int *len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len = len; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  __auto_type local_counted_ptr_deref = *ptr; // expected-error{{passing '__counted_by' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_len_deref = *len;
  __auto_type local_counted_ptr_addrof = &ptr; // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
}

void TestOutSizedBy(void *__sized_by(*len) *ptr, unsigned *len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{pointer with '__sized_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len = len;  // expected-error{{variable referred to by '__sized_by' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  __auto_type local_counted_ptr_deref = *ptr; // expected-error{{passing '__sized_by' pointer as __auto_type initializer is not yet supported}}
}

void TestCountedByOrNull(int *__counted_by_or_null(len) ptr, int len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{passing '__counted_by_or_null' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_len = len;
  __auto_type local_counted_ptr_addrof = &ptr; // expected-error{{pointer with '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len_addrof = &len;         // expected-error{{variable referred to by '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
}

void TestSizedByOrNull(void *__sized_by_or_null(len) ptr, unsigned len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{passing '__sized_by_or_null' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_len = len;
  __auto_type local_counted_ptr_addrof = &ptr; // expected-error{{pointer with '__sized_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len_addrof = &len;         // expected-error{{variable referred to by '__sized_by_or_null' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
}

void TestOutCountedByOrNull(int *__counted_by_or_null(*len) *ptr, int *len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{pointer with '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len = len; // expected-error{{variable referred to by '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  __auto_type local_counted_ptr_deref = *ptr; // expected-error{{passing '__counted_by_or_null' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_len_deref = *len;
  __auto_type local_counted_ptr_addrof = &ptr; // expected-error{{pointer with '__counted_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
}

void TestOutSizedByOrNull(void *__sized_by_or_null(*len) *ptr, unsigned *len) {
  __auto_type local_counted_ptr = ptr; // expected-error{{pointer with '__sized_by_or_null' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len = len;  // expected-error{{variable referred to by '__sized_by_or_null' cannot be pointed to by any other variable; exception is when the pointer is passed as a compatible argument to a function}}
  __auto_type local_counted_ptr_deref = *ptr; // expected-error{{passing '__sized_by_or_null' pointer as __auto_type initializer is not yet supported}}
}

void TestOutCountedByArray(int (*ptr)[__counted_by(*len)], int *len) { // expected-error{{pointer to incomplete __counted_by array type 'int[]' not allowed; did you mean to use a nested pointer type?}}
  __auto_type local_counted_ptr = ptr; // expected-error{{array with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
  __auto_type local_len = len; // expected-error{{variable referred to by '__counted_by' cannot be pointed to by any other variable}}
  __auto_type local_counted_ptr_deref = *ptr; // expected-error{{passing '__counted_by' pointer as __auto_type initializer is not yet supported}}
  __auto_type local_len_deref = *len;
  __auto_type local_counted_ptr_addrof = &ptr; // expected-error{{array with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
}


struct CountedStruct {
    char * __counted_by(n) buf;
    int n;
};

void foo(struct CountedStruct * __single p) {
    __auto_type ebuf = p->buf; // expected-error{{passing '__counted_by' pointer as __auto_type initializer is not yet supported}}
    __auto_type *__single eptr = &(p->buf); // expected-error{{pointer with '__counted_by' cannot be pointed to by any other variable; exception is when the variable is passed as a compatible argument to a function}}
}

void baz(void * __unsafe_indexable p);

void bar(char * p) {
    char * implicit_bidi = p;
    baz(
        (void*)({
            __auto_type pointer_to_bidi = &implicit_bidi;
            *pointer_to_bidi;
        })
    );
}

struct FAMStruct {
    int n;
    char buf[__counted_by(n)];
};

void foo_fam(struct FAMStruct * __single p) {
    __auto_type ebuf = p->buf; // expected-error{{passing '__counted_by' pointer as __auto_type initializer is not yet supported}}
    __auto_type *__single eptr = &(p->buf); // expected-error{{cannot take address of incomplete __counted_by array}}
                                            // expected-note@-1{{remove '&' to get address as 'char *' instead of 'char (*)[__counted_by(n)]'}}
}
