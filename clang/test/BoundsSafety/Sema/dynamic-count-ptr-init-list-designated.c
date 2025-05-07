
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s

#include <ptrcheck.h>

#define INT_MIN (-2147483648)

struct S { int len; int *__counted_by(len) ptr; };

// expected-note@+1{{consider adding '__counted_by(2)' to 'a'}}
void TestCountedBy(int *a) {
  // expected-note@+1{{'arr' declared here}}
	int arr[] = {1,2,3};
  // expected-error@+1{{implicitly initializing 's1.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 4 with null always fails}}
  struct S s1 = { .len = 4 };
  struct S s2 = { .len = 0 }; // ok
  // expected-error@+1{{negative count value of -1 for 's3.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s3 = { .len = -1 };
  // expected-error@+1{{initializing 's4.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 4 with null always fails}}
  struct S s4 = { .ptr = 0, .len = 4 };
  struct S s5 = { .len = 0, .ptr = 0 }; // ok
  // expected-error@+1{{negative count value of -1 for 's6.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s6 = { .len = -1, .ptr = 0 };
  // expected-error@+1{{initializing 's7.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 4 with array 'arr' (which has 3 elements) always fails}}
  struct S s7 = { 4, .ptr = arr };
  struct S s8 = { .ptr = arr, .len = 3 }; // ok
  // expected-error@+1{{negative count value of -2147483648 for 's9.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s9 = { .len = INT_MIN, .ptr = arr };
  // expected-error@+1{{initializing 's10.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 2 with 'int *__single' always fails}}
  struct S s10 = { .len = 2, .ptr = a };
  struct S s11 = { .len = 1, .ptr = a };
  // expected-error@+1{{negative count value of -2147483648 for 's12.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s12 = { .len = INT_MIN, .ptr = a };
  struct S s13 = { .ptr = 0 }; // ok
  // expected-warning@+1{{possibly initializing 's14.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}
  struct S s14 = { .ptr = arr };
  // expected-warning@+1{{possibly initializing 's15.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and implicit count value of 0 with non-null, which creates a non-dereferenceable pointer; explicitly set count value to 0 to remove this warning}}
  struct S s15 = { .ptr = a };
}

struct U { int len; void *__sized_by(len - 1) ptr; };

void TestSizedByP1(char *a) {
	char arr[] = {1,2,3};
  // expected-error@+1{{implicitly initializing 's1.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single') and size value of 3 with null always fails}}
  struct U s1 = { .len = 4 };
  // expected-error@+1{{negative size value of -1 for 's2.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s2 = { .len = 0 };
  // expected-error@+1{{negative size value of -1 for 's4.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s4 = { .ptr = 0, .len = 0 };
  struct U s5 = { .len = 4, .ptr = arr }; // ok
  // expected-error@+1{{negative size value of -1 for 's6.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s6 = { .len = 0, .ptr = arr };
  struct U s7 = { .len = 2, .ptr = a };
  struct U s8 = { .len = 1, .ptr = a };
  // expected-error@+1{{negative size value of -1 for 's9.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s9 = { .ptr = a, .len = 0 };
  // expected-error@+1{{negative size value of -1 for 's10.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s10;
  // expected-error@+1{{negative size value of -1 for 's11[0].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s11[2];
  // expected-error@+1{{negative size value of -1 for 's12[0].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s12[1] = { {.len = 0} };
  struct U s13[1] = { {.len = 1} }; // ok
  // expected-error@+1{{negative size value of -1 for 's14[1].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s14[2] = { {.len = 1} };

  // expected-error@+1{{negative size value of -1 for 's15.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s15 = { .ptr = 0 };
  // expected-error@+1{{negative size value of -1 for 's16.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s16 = { .ptr = arr };
  // expected-error@+1{{negative size value of -1 for 's17.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s17 = { .ptr = a };
  // expected-error@+1{{negative size value of -1 for 's19[0].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s19[1] = { {.ptr = 0} };
  // expected-error@+2{{negative size value of -1 for 's20[0].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  // expected-error@+1{{negative size value of -1 for 's20[1].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s20[2] = { {.ptr = arr} };
}

struct T { void *__sized_by(len + 1) ptr; int len; };

void TestSizedByPP1PtrFirst(void) {
  // expected-note@+1{{'arr' declared here}}
  char arr[] = {1, 2, 3};
  // expected-error@+1{{initializing 's1.ptr' of type 'void *__single __sized_by(len + 1)' (aka 'void *__single') and size value of 1 with null always fails}}
  struct T s1 = { .ptr = 0 };
  struct T s2 = { .ptr = arr }; // ok
  // expected-error@+1{{implicitly initializing 's3[1].ptr' of type 'void *__single __sized_by(len + 1)' (aka 'void *__single') and size value of 1 with null always fails}}
  struct T s3[4] = { {.len = -1} };
  struct T s4[1] = { {.len = -1} }; // ok
  // expected-error@+1{{implicitly initializing 's5[1].ptr' of type 'void *__single __sized_by(len + 1)' (aka 'void *__single') and size value of 1 with null always fails}}
  struct T s5[4] = { {.len = -1, .ptr = 0} };
  struct T s6[1] = { {.len = -1, .ptr = 0} }; // ok
  // expected-error@+1{{initializing 's7.ptr' of type 'void *__single __sized_by(len + 1)' (aka 'void *__single') and size value of 4 with array 'arr' (which has 3 bytes) always fails}}
  struct T s7 = { .ptr = arr, 3 };
}
