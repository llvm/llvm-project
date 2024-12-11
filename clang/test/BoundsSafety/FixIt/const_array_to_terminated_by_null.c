
// RUN: cp %s %t
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits -verify %t
// RUN: not %clang_cc1 -fsyntax-only -fbounds-safety -fdiagnostics-parseable-fixits %t > %t.cc_out 2> %t.cc_out
// RUN: FileCheck %s --input-file=%t.cc_out

#include <ptrcheck.h>

// expected-note@+1 10{{passing argument to parameter 'path' here}}
void method_with_single_const_param(const char *path); 

// expected-note@+1 10{{passing argument to parameter 'path' here}}
void method_with_single_null_terminated_param(char * __null_terminated path);

void test_single_invocation_local_variable() {

  // expected-note@+2{{consider adding '__null_terminated' to 'array_of_chars'}}
  // CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:23-[[@LINE+1]]:23}:"__null_terminated"
  char array_of_chars[] = "foo";

  // CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+2]]:3-[[@LINE+2]]:3}:"const "
  // expected-note@+1{{consider adding 'const' to 'array_of_chars2'}}
  char array_of_chars2[] = "foo";

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[4]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars);

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1 {{passing 'char[4]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars2);
}

// expected-note@+2{{consider adding '__null_terminated' to 'array_of_chars'}}
// CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:21-[[@LINE+1]]:21}:"__null_terminated"
char array_of_chars[] = "foo";

// CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+2]]:22-[[@LINE+2]]:22}:"__null_terminated"
// expected-note@+1{{consider adding '__null_terminated' to 'array_of_chars2'}}
char array_of_chars2[] = "foo";

void test_single_invocation_global_variable() {

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[4]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars);

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1 {{passing 'char[4]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars2);
}

void test_explicit_const_size_arrays() {
  // expected-note@+2{{consider adding '__null_terminated' to 'array_of_chars'}}
  // CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:23-[[@LINE+1]]:23}:"__null_terminated "
  char array_of_chars[20] = "foo";

  // CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+2]]:3-[[@LINE+2]]:3}:"const "
  // expected-note@+1{{consider adding 'const' to 'array_of_chars2'}}
  char array_of_chars2[20] = "foobar";

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[20]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars);

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1 {{passing 'char[20]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars2);
}

void test_initialized_local_arrays_without_null_termination() {
  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:23-[[@LINE+1]]:23}:"__null_terminated "
  char array_of_chars[] = {'f', 'o', 'o'};

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[3]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars);

  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const "
  char array_of_chars2[] = {'f', 'o', 'o', 'b', 'a', 'r'};

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1 {{passing 'char[6]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars2);
}

void test_initialized_local_arrays_with_null_termination() {
  // expected-note@+2{{consider adding '__null_terminated' to 'array_of_chars'}}
  // CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:23-[[@LINE+1]]:23}:"__null_terminated"
  char array_of_chars[] = {'f', 'o', 'o', '\0'};

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[4]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars);

  // CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+2]]:3-[[@LINE+2]]:3}:"const "
  // expected-note@+1{{consider adding 'const' to 'array_of_chars2'}}
  char array_of_chars2[] = {'f', 'o', 'o', 'b', 'a', 'r', '\0'};

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1 {{passing 'char[7]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars2);

  char array_of_chars3[6][2] = {{'f'}, {'o'}, {'x', 'y'}, {'b', 'a'}, {'r', '\0'}};
  
  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[6][2]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars3);
  
  char array_of_chars4[6][2] = {{'f'}, {'o'}, {'x', 'y'}, {'b', 'a'}, {'r', '\0'}};

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1 {{passing 'char[6][2]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars4);
 
}

typedef char my_array_t[4];
// expected-note@+2{{consider adding '__null_terminated' to 'typedef_array1'}}
// CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__null_terminated "
my_array_t typedef_array1 = "foo";

// expected-note@+2{{consider adding '__null_terminated' to 'typedef_array2'}}
// CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:12-[[@LINE+1]]:12}:"__null_terminated "
my_array_t typedef_array2 = "bar";

void test_typedefed_arrays() {
  // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1{{passing 'my_array_t' (aka 'char[4]') to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(typedef_array1);

  // expected-note@+3{{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2{{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-error@+1{{passing 'my_array_t' (aka 'char[4]') to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(typedef_array2);

}

// expected-note@+2{{consider adding '__null_terminated' to 'array_of_chars123'}}
// CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:24-[[@LINE+1]]:24}:"__null_terminated "
char array_of_chars123[10];

// CHECK-DAG: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+2]]:24-[[@LINE+2]]:24}:"__null_terminated "
// expected-note@+1{{consider adding '__null_terminated' to 'array_of_chars345'}}
char array_of_chars345[6];

void test_uninitialized_global_arrays() {

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[10]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars123);


  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}} 
  // expected-error@+1 {{passing 'char[6]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars345);
} 

void test_no_fixits() {
  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+2]]:3-[[@LINE+2]]:3}:"const "
  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:30-[[@LINE+1]]:30}:"__null_terminated "
  const char array_of_chars1[] = "foo";
 
  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const " 
  char array_of_chars2[__null_terminated] = "bar";

  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const "
  char array_of_chars3[__null_terminated 20] = "bar";

  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const "
  char array_of_chars4[3] = "foo";

  // CHECK-NOT: fix-it:"{{.+}}const_array_to_terminated_by_null.c.tmp":{[[@LINE+1]]:3-[[@LINE+1]]:3}:"const "
  char array_of_chars5[3] = {};

  // expected-warning@+1 {{passing 'const char[4]' to parameter of type 'char *__single __terminated_by(0)' (aka 'char *__single') discards qualifiers}}
  method_with_single_null_terminated_param(array_of_chars1);

  method_with_single_const_param(array_of_chars1); 

  method_with_single_null_terminated_param(array_of_chars2);
  
  method_with_single_const_param(array_of_chars2);
  
  method_with_single_null_terminated_param(array_of_chars3);
  
  method_with_single_const_param(array_of_chars3);

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[3]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_const_param(array_of_chars4);

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[3]' to parameter of incompatible type 'char *__single __terminated_by(0)' (aka 'char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}
  method_with_single_null_terminated_param(array_of_chars4);

  // expected-note@+3 {{consider using '__unsafe_null_terminated_from_indexable()' with a pointer to the null terminator to perform this conversion. Note this performs the conversion in constant time}}
  // expected-note@+2 {{consider using '__unsafe_null_terminated_from_indexable()' to perform this conversion. Note this performs a linear scan of memory to find the null terminator}}
  // expected-error@+1 {{passing 'char[3]' to parameter of incompatible type 'const char *__single __terminated_by(0)' (aka 'const char *__single') is an unsafe operation; use '__unsafe_null_terminated_from_indexable()' or '__unsafe_forge_null_terminated()' to perform this conversion}}   
  method_with_single_const_param(array_of_chars4);

}
