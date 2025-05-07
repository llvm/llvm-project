
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
  struct S s1 = { 4 };
  struct S s2 = { 0 }; // ok
  // expected-error@+1{{negative count value of -1 for 's3.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s3 = { -1 };
  // expected-error@+1{{initializing 's4.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 4 with null always fails}}
  struct S s4 = { 4, 0 };
  struct S s5 = { 0, 0 }; // ok
  // expected-error@+1{{negative count value of -1 for 's6.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s6 = { -1, 0 };
  // expected-error@+1{{initializing 's7.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 4 with array 'arr' (which has 3 elements) always fails}}
  struct S s7 = { 4, arr };
  struct S s8 = { 3, arr }; // ok
  // expected-error@+1{{negative count value of -2147483648 for 's9.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s9 = { INT_MIN, arr };
  // expected-error@+1{{initializing 's10.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single') and count value of 2 with 'int *__single' always fails}}
  struct S s10 = { 2, a };
  struct S s11 = { 1, a };
  // expected-error@+1{{negative count value of -2147483648 for 's12.ptr' of type 'int *__single __counted_by(len)' (aka 'int *__single')}}
  struct S s12 = { INT_MIN, a };
}

struct V { unsigned len; void *__sized_by(len) ptr; };

// expected-note@+1{{consider adding '__sized_by(2)' to 'a'}}
void TestSizedBy(char *a) {
  // expected-note@+1{{'arr' declared here}}
	char arr[] = {1,2,3};
  // expected-error@+1{{implicitly initializing 's1.ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single') and size value of 4 with null always fails}}
  struct V s1 = { 4 };
  struct V s2 = { 0 };
  // expected-error@+1{{initializing 's3.ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single') and size value of 4 with null always fails}}
  struct V s3 = { 4, 0 };
  struct V s4 = { 0, 0 };
  // expected-error@+1{{initializing 's5.ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single') and size value of 4 with array 'arr' (which has 3 bytes) always fails}}
  struct V s5 = { 4, arr };
  struct V s6 = { 3, arr };
  // expected-error@+1{{initializing 's7.ptr' of type 'void *__single __sized_by(len)' (aka 'void *__single') and size value of 2 with 'char *__single' and pointee of size 1 always fails}}
  struct V s7 = { 2, a };
  struct V s8 = { 1, a };
}

struct S2 { int len; int *__counted_by_or_null(len) ptr; };

void TestCountedByOrNull(int *a) {
  // expected-note@+1{{'arr' declared here}}
	int arr[] = {1,2,3};
  struct S2 s1 = { 4 }; // ok
  struct S2 s2 = { 0 }; // ok
  struct S2 s3 = { -1 }; // ok
  struct S2 s4 = { 4, 0 }; // ok
  struct S2 s5 = { 0, 0 }; // ok
  struct S2 s6 = { -1, 0 }; // ok
  // expected-error@+1{{initializing 's7.ptr' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') and count value of 4 with array 'arr' (which has 3 elements) always fails}}
  struct S2 s7 = { 4, arr };
  struct S2 s8 = { 3, arr }; // ok
  // expected-error@+1{{negative count value of -2147483648 for 's9.ptr' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single')}}
  struct S2 s9 = { INT_MIN, arr };
  struct S2 s10 = { 2, a };
  struct S2 s11 = { 1, a };
  // expected-error@+1{{possibly initializing 's12.ptr' of type 'int *__single __counted_by_or_null(len)' (aka 'int *__single') and count value of -2147483648 with non-null; explicitly initialize null to remove this warning}}
  struct S2 s12 = { INT_MIN, a };
}

struct V2 { unsigned len; void *__sized_by_or_null(len) ptr; };

void TestSizedByOrNull(char *a) {
  // expected-note@+1{{'arr' declared here}}
	char arr[] = {1,2,3};
  struct V2 s1 = { 4 }; // ok
  struct V2 s2 = { 0 }; // ok
  struct V2 s3 = { 4, 0 }; // ok
  struct V2 s4 = { 0, 0 }; // ok
  // expected-error@+1{{initializing 's5.ptr' of type 'void *__single __sized_by_or_null(len)' (aka 'void *__single') and size value of 4 with array 'arr' (which has 3 bytes) always fails}}
  struct V2 s5 = { 4, arr };
  struct V2 s6 = { 3, arr };
  struct V2 s7 = { 2, a };
  struct V2 s8 = { 1, a };
}

struct U { int len; void *__sized_by(len - 1) ptr; };

void TestSizedByP1(char *a) {
	char arr[] = {1,2,3};
  // expected-error@+1{{implicitly initializing 's1.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single') and size value of 3 with null always fails}}
  struct U s1 = { 4 };
  // expected-error@+1{{negative size value of -1 for 's2.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s2 = { 0 };
  // expected-error@+1{{negative size value of -1 for 's4.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s4 = { 0, 0 };
  struct U s5 = { 4, arr }; // ok
  // expected-error@+1{{negative size value of -1 for 's6.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s6 = { 0, arr };
  struct U s7 = { 2, a };
  struct U s8 = { 1, a };
  // expected-error@+1{{negative size value of -1 for 's9.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s9 = { 0, a };
  // expected-error@+1{{negative size value of -1 for 's10.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s10;
  // expected-error@+1{{negative size value of -1 for 's11[0].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s11[2];
  // expected-error@+1{{negative size value of -1 for 's12[0].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s12[1] = { 0 };
  struct U s13[1] = { 1 }; // ok
  // expected-error@+1{{negative size value of -1 for 's14[1].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct U s14[2] = { 1 };
}

struct U2 { int len; void *__sized_by_or_null(len - 1) ptr; };

void TestSizedByOrNullP1(char *a) {
	char arr[] = {1,2,3};
  struct U2 s1 = { 4 }; // ok
  struct U2 s2 = { 0 }; // ok
  struct U2 s4 = { 0, 0 }; // ok
  struct U2 s5 = { 4, arr }; // ok
  // expected-error@+1{{negative size value of -1 for 's6.ptr' of type 'void *__single __sized_by_or_null(len - 1)' (aka 'void *__single')}}
  struct U2 s6 = { 0, arr };
  struct U2 s7 = { 2, a };
  struct U2 s8 = { 1, a };
  // expected-error@+1{{possibly initializing 's9.ptr' of type 'void *__single __sized_by_or_null(len - 1)' (aka 'void *__single') and size value of -1 with non-null; explicitly initialize null to remove this warning}}
  struct U2 s9 = { 0, a };
  struct U2 s10; // ok
  struct U2 s11[2]; // ok
  struct U2 s12[1] = { 0 }; // ok
  struct U2 s13[1] = { 1 }; // ok
  struct U2 s14[2] = { 1 }; // ok
}

struct T { void *__sized_by(len - 1) ptr; int len; };

void TestSizedByP1PtrFirst() {
  char arr[] = {1, 2, 3};
  // expected-error@+1{{negative size value of -1 for 's1.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct T s1 = { 0 };
  // expected-error@+1{{negative size value of -1 for 's2.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct T s2 = { arr };
  // expected-error@+1{{negative size value of -1 for 's3[1].ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct T s3[4] = { arr, 4};
  struct T s4[1] = { arr, 4}; // ok
}

struct T2 { void *__sized_by_or_null(len - 1) ptr; int len; };

void T2estSizedByOrNullP1PtrFirst() {
  char arr[] = {1, 2, 3};
  struct T2 s1 = { 0 }; // ok
  // expected-error@+1{{negative size value of -1 for 's2.ptr' of type 'void *__single __sized_by_or_null(len - 1)' (aka 'void *__single')}}
  struct T2 s2 = { arr };
  struct T2 s3[4] = { arr, 4}; // ok
  struct T2 s4[1] = { arr, 4}; // ok
}

struct W { void *__sized_by(len + 1) ptr; int len; };

void TestSizedByPP1PtrFirst() {
  // expected-note@+1{{'arr' declared here}}
  char arr[] = {1, 2, 3};
  // expected-error@+1{{initializing 's1.ptr' of type 'void *__single __sized_by(len + 1)' (aka 'void *__single') and size value of 1 with null always fails}}
  struct W s1 = { 0 };
  struct W s2 = { arr };
  // expected-error@+1{{implicitly initializing 's3[1].ptr' of type 'void *__single __sized_by(len + 1)' (aka 'void *__single') and size value of 1 with null always fails}}
  struct W s3[4] = { 0, -1 };
  struct W s4[1] = { 0, -1 }; // ok
  // expected-error@+1{{initializing 's5.ptr' of type 'void *__single __sized_by(len + 1)' (aka 'void *__single') and size value of 4 with array 'arr' (which has 3 bytes) always fails}}
  struct W s5 = { arr, 3 };
}

struct W2 { void *__sized_by_or_null(len + 1) ptr; int len; };

void TestSizedByPP1PtrFirstOrNull() {
  // expected-note@+1{{'arr' declared here}}
  char arr[] = {1, 2, 3};
  struct W2 s1 = { 0 }; // ok
  struct W2 s2 = { arr }; // ok
  struct W2 s3[4] = { 0, -1 }; // ok
  struct W2 s4[1] = { 0, -1 }; // ok
  // expected-error@+1{{initializing 's5.ptr' of type 'void *__single __sized_by_or_null(len + 1)' (aka 'void *__single') and size value of 4 with array 'arr' (which has 3 bytes) always fails}}
  struct W2 s5 = { arr, 3 };
}

// TODO: rdar://76377847
void TestAssignAfterDeclNoError(void) {
  char arr[] = {1, 2, 3};
  // expected-error@+1{{negative size value of -1 for 's.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct T s;
  s.ptr = arr;
  s.len = 4;
}

void TestAssignAfterDeclError(void) {
  char arr[] = {1, 2, 3};
  // expected-error@+1{{negative size value of -1 for 's.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct T s;
  (void)s.ptr;
  s.ptr = arr;
  s.len = 4;
}

void T2estAssignAfterDeclNoErrorOrNull(void) {
  char arr[] = {1, 2, 3};
  struct T2 s;
  s.ptr = arr;
  s.len = 4;
}

void T2estAssignAfterDeclErrorOrNull(void) {
  char arr[] = {1, 2, 3};
  struct T2 s;
  (void)s.ptr;
  s.ptr = arr;
  s.len = 4;
}

struct NestedS {
  struct S s;
};

struct ReallyNestedS {
  struct NestedS nested_s[2];
};

// struct S has __counted_by(len), so zero-init is OK.
void TestImplicitNestedS(void) {
  struct NestedS nested_s;
  struct NestedS nested_s_arr[3];

  struct ReallyNestedS really_nested_s;
  struct ReallyNestedS really_nested_s_arr[3];
}

struct NestedU {
  struct U u;
};

struct ReallyNestedU {
  struct NestedU nested_u[2];
};

// struct U has __sized_by(len - 1), so zero-init is BAD.
void TestImplicitNestedU(void) {
  // expected-error@+1{{negative size value of -1 for 'nested_u.u.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct NestedU nested_u;

  // expected-error@+1{{negative size value of -1 for 'nested_u_arr[0].u.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct NestedU nested_u_arr[3];

  // expected-error@+1{{negative size value of -1 for 'really_nested_u.nested_u[0].u.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct ReallyNestedU really_nested_u;

  // expected-error@+1{{negative size value of -1 for 'really_nested_u_arr[0].nested_u[0].u.ptr' of type 'void *__single __sized_by(len - 1)' (aka 'void *__single')}}
  struct ReallyNestedU really_nested_u_arr[3];
}
