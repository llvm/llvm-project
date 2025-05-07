
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify=bounds-safety %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify=bounds-safety %s

#include <ptrcheck.h>

void foo(int *__counted_by(len) ptr, int len) {}

// bounds-safety-note@+3{{consider adding '__counted_by(len)' to 'ptr'}}
// bounds-safety-note@+2{{consider adding '__counted_by(2)' to 'ptr'}}
// bounds-safety-note@+1{{consider adding '__counted_by(3)' to 'ptr'}}
void bar(int *ptr, int len) {
  int *__indexable ptr_bound;
  // bounds-safety-note@+2{{count passed here}}
  // bounds-safety-warning@+1{{count value is not statically known: passing 'int *__single' to parameter 'ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') is invalid for any count other than 0 or 1}}
  foo(ptr, len);
  foo(ptr, 1);
  const int one = 1;
  foo(ptr, one);
  const int zero = 0;
  foo(ptr, zero);
  const int two = 2;
  // bounds-safety-error@+1{{passing 'int *__single' to parameter 'ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') with count value of 2 always fails}}
  foo(ptr, two);
  const int neg = -1;
  // bounds-safety-error@+1{{negative count value of -1 for 'ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  foo(ptr, neg);

  foo(ptr_bound, len);
  foo(ptr_bound, 1);
  foo(ptr_bound, zero);
  foo(ptr_bound, two);
  // bounds-safety-error@+1{{negative count value of -1 for 'ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  foo(ptr_bound, neg);

  int local_len;
  // bounds-safety-warning@+1{{possibly initializing 'local_ptr' of type 'int *__single __counted_by(local_len)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}
  int *__counted_by(local_len) local_ptr = ptr;
  int *__counted_by(1) local_ptr2 = ptr;
  // bounds-safety-error@+1{{initializing 'local_ptr3' of type 'int *__single __counted_by(2)' (aka 'int *__single') and count value of 2 with 'int *__single' always fails}}
  int *__counted_by(2) local_ptr3 = ptr;
  const int three = 3;
  // bounds-safety-error@+1{{initializing 'local_ptr4' of type 'int *__single __counted_by(3)' (aka 'int *__single') and count value of 3 with 'int *__single' always fails}}
  int *__counted_by(three) local_ptr4 = ptr;
  const int neg2 = -1;
  // bounds-safety-error@+1{{negative count value of -1 for 'local_ptr5' of type 'int *__single __counted_by(-1)' (aka 'int *__single')}}
  int *__counted_by(neg2) local_ptr5 = ptr;
}

void baz(int *__counted_by(len) ptr1, int *__counted_by(1) ptr2, int len) {
  int *__single sp;
  int *__unsafe_indexable up;
  // bounds-safety-warning@+1{{count value is not statically known: assigning to 'ptr1' of type 'int *__single __counted_by(len)' (aka 'int *__single') from 'int *__single' is invalid for any count other than 0 or 1}}
  ptr1 = sp;
  // bounds-safety-note@+1{{count assigned here}}
  len = len;

  ptr2 = sp;
  ptr2 = up; // bounds-safety-error{{assigning to 'int *__single __counted_by(1)' (aka 'int *__single') from incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
}

void foo_nullable(int *__counted_by_or_null(len) ptr, int len) {}

// bounds-safety-note@+1{{consider adding '__counted_by_or_null(len)' to 'ptr'}}
void bar_nullable(int *ptr, int len) {
  int *__indexable ptr_bound;
  // bounds-safety-note@+2{{count passed here}}
  // bounds-safety-warning@+1{{count value is not statically known: passing 'int *__single' to parameter 'ptr' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') is invalid for any count other than 0 or 1 unless the pointer is null}}
  foo_nullable(ptr, len);
  foo_nullable(ptr, 1);
  const int one = 1;
  foo_nullable(ptr, one);
  const int zero = 0;
  foo_nullable(ptr, zero);
  const int two = 2;
  foo_nullable(ptr, two);
  const int neg = -1;
  // bounds-safety-error@+1{{possibly passing non-null to parameter 'ptr' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') with count value of -1; explicitly pass null to remove this warning}}
  foo_nullable(ptr, neg);

  foo_nullable(ptr_bound, len);
  foo_nullable(ptr_bound, 1);
  foo_nullable(ptr_bound, zero);
  foo_nullable(ptr_bound, two);
  // bounds-safety-error@+1{{possibly passing non-null to parameter 'ptr' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') with count value of -1; explicitly pass null to remove this warning}}
  foo_nullable(ptr_bound, neg);

  int local_len;
  // bounds-safety-warning@+1{{possibly initializing 'local_ptr' of type 'int *__single __counted_by_or_null(local_len)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}
  int *__counted_by_or_null(local_len) local_ptr = ptr;
  int *__counted_by_or_null(1) local_ptr2 = ptr;
  int *__counted_by_or_null(2) local_ptr3 = ptr;
  const int three = 3;
  int *__counted_by_or_null(three) local_ptr4 = ptr;
  const int neg2 = -1;
  // bounds-safety-error@+1{{possibly initializing 'local_ptr5' of type 'int *__single __counted_by_or_null(-1)' (aka 'int *__single') and count value of -1 with non-null; explicitly initialize null to remove this warning}}
  int *__counted_by_or_null(neg2) local_ptr5 = ptr;
}

void baz_nullable(int *__counted_by_or_null(len) ptr1, int *__counted_by_or_null(1) ptr2, int len) {
  int *__single sp;
  int *__unsafe_indexable up;
  // bounds-safety-warning@+1{{count value is not statically known: assigning to 'ptr1' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') from 'int *__single' is invalid for any count other than 0 or 1 unless the pointer is null}}
  ptr1 = sp;
  // bounds-safety-note@+1{{count assigned here}}
  len = len;

  ptr2 = sp;
  ptr2 = up; // bounds-safety-error{{assigning to 'int *__single __counted_by_or_null(1)' (aka 'int *__single') from incompatible type 'int *__unsafe_indexable' casts away '__unsafe_indexable' qualifier}}
}
