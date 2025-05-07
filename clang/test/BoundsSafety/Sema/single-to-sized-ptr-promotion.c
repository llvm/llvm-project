
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -Wno-bounds-safety-single-to-indexable-bounds-truncated -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>
void foo(int *__sized_by(len) ptr, int len) {}
void foo_void(void *__sized_by(len) ptr, int len) {}
void foo_fixed(void *__sized_by(8) ptr) {}

// expected-note@+4{{consider adding '__sized_by(len)' to 'ptr'}}
// expected-note@+3{{consider adding '__sized_by(8)' to 'ptr'}}
// expected-note@+2{{consider adding '__sized_by(5)' to 'ptr'}}
// expected-note@+1{{consider adding '__sized_by(12)' to 'ptr'}}
void bar(int *ptr, int len) {
    int *ptr_auto_bound;
    // expected-note@+2{{size passed here}}
    // expected-warning@+1{{size value is not statically known: passing 'int *__single' to parameter 'ptr' of type 'int *__single __sized_by(len)' (aka 'int *__single') is invalid for any size greater than 4}}
    foo(ptr, len);
    foo(ptr, 1);
    foo(ptr, sizeof(int));
    const int four = 1 * sizeof(int);
    foo(ptr, four);
    const int zero = 0;
    foo(ptr, zero);
    const int eight = 2 * sizeof(int);
    foo(ptr, eight); // expected-error{{passing 'int *__single' with pointee of size 4 to parameter 'ptr' of type 'int *__single __sized_by(len)' (aka 'int *__single') with size value of 8 always fails}}
    const int neg = -1;
    foo(ptr, neg); // expected-error{{negative size value of -1 for 'ptr' of type 'int *__single __sized_by(len)' (aka 'int *__single')}}
    foo_fixed(ptr); // expected-error{{passing 'int *__single' with pointee of size 4 to parameter 'ptr' of type 'void *__single __sized_by(8)' (aka 'void *__single') with size value of 8 always fails}}

    foo_void(ptr, zero);
    foo_void(ptr, four);
    foo_void(ptr, eight); // expected-error{{passing 'int *__single' with pointee of size 4 to parameter 'ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single') with size value of 8 always fails}}
    foo_void(ptr, neg); // expected-error{{negative size value of -1 for 'ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single')}}

    foo(ptr_auto_bound, len);
    foo(ptr_auto_bound, 1);
    foo(ptr_auto_bound, zero);
    foo(ptr_auto_bound, eight);
    foo(ptr_auto_bound, neg); // expected-error{{negative size value of -1 for 'ptr' of type 'int *__single __sized_by(len)' (aka 'int *__single')}}

    int local_len;
    // expected-warning@+1{{possibly initializing 'local_ptr' of type 'int *__single __sized_by(local_len)' (aka 'int *__single') and implicit size value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set size value to 0 to remove this warning}}
    int *__sized_by(local_len) local_ptr = ptr;
    int *__sized_by(4) local_ptr2 = ptr;
    // expected-error@+1{{initializing 'local_ptr3' of type 'int *__single __sized_by(5)' (aka 'int *__single') and size value of 5 with 'int *__single' and pointee of size 4 always fails}}
    int *__sized_by(5) local_ptr3 = ptr;
    const int three = 3 * sizeof(int);
    // expected-error@+1{{initializing 'local_ptr4' of type 'int *__single __sized_by(12)' (aka 'int *__single') and size value of 12 with 'int *__single' and pointee of size 4 always fails}}
    int *__sized_by(three) local_ptr4 = ptr;
    const int neg2 = -1;
    // expected-error@+1{{negative size value of -1 for 'local_ptr5' of type 'int *__single __sized_by(-1)' (aka 'int *__single')}}
    int *__sized_by(neg2) local_ptr5 = ptr;
}

void baz(int *__sized_by(len) ptr1, int *__sized_by(4) ptr2, int len) {
    int *__single sp;
    int *__unsafe_indexable up;
    // expected-warning@+1{{size value is not statically known: assigning to 'ptr1' of type 'int *__single __sized_by(len)' (aka 'int *__single') from 'int *__single' is invalid for any size greater than 4}}
    ptr1 = sp;
    // expected-note@+1{{size assigned here}}
    len = len;
    ptr1 = up; // expected-error-re{{assigning to {{.+}} from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}

    ptr2 = sp;
    ptr2 = up; // expected-error-re{{assigning to {{.+}} from incompatible type '{{.+}}*__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
}

struct qux;

void quux(void *__sized_by(42) ptr1,
          void *__sized_by(0) ptr2,
          void *__sized_by(len) ptr3, int len,
          struct qux *__single qux) {
    // expected-error@+1{{assigning to 'ptr1' of type 'void *__single __sized_by(42)' (aka 'void *__single') with size value of 42 from 'struct qux *__single' with pointee of size 0 always fails}}
    ptr1 = qux;

    ptr2 = qux; // ok

    // expected-error@+1{{assigning to 'ptr3' of type 'void *__single __sized_by(len)' (aka 'void *__single') with size value of 42 from 'struct qux *__single' with pointee of size 0 always fails}}
    ptr3 = qux;
    len = 42;
}

struct unsized;

void external(struct unsized *__sized_by(size) p, int size);

void convert_unsized(struct unsized *__single p, int size) {
  // expected-error@+1{{argument of '__sized_by' attribute cannot refer to declaration of a different lifetime}}
  struct unsized *__single __sized_by(size) a = p; // requires explicit __single because of rdar://112196572
  // expected-warning@+1{{size value is not statically known: passing 'struct unsized *__single' to parameter 'p' of type 'struct unsized *__single __sized_by(size)' (aka 'struct unsized *__single') is invalid for any size greater than 0}}
  external(p, size); // expected-note{{size passed here}}
}

void convert_unsized2(struct unsized *__single p, int size) {
  // expected-note@+1{{size initialized here}}
  int s = size;
  // expected-warning@+1{{size value is not statically known: initializing 'a' of type 'struct unsized *__single __sized_by(s)' (aka 'struct unsized *__single') with 'struct unsized *__single' is invalid for any size greater than 0}}
  struct unsized *__single __sized_by(s) a = p; // requires explicit __single because of rdar://112196572
  // expected-warning@+1{{size value is not statically known: passing 'struct unsized *__single' to parameter 'p' of type 'struct unsized *__single __sized_by(size)' (aka 'struct unsized *__single') is invalid for any size greater than 0}}
  external(p, size); // expected-note{{size passed here}}
}
