
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

// const arguments
unsigned global_count = 2;
const unsigned const_global_count = 0;
extern const unsigned extern_const_global_count;

enum {
  enum_count = 10
};

struct struct_enum_count {
  int *__counted_by(enum_count) buf;
};

struct struct_const_global_count {
  int *__counted_by(const_global_count) buf;
};

void fun_const_global_count(int *__counted_by(const_global_count) arg);

void test_local_const_global_count() {
  int *__counted_by(const_global_count) local;
}

struct struct_extern_const_global_count {
  int *__counted_by(extern_const_global_count) buf;
};

void fun_extern_const_global_count(int *__counted_by(extern_const_global_count) arg);

void test_local_extern_const_global_count() {
  int *__counted_by(extern_const_global_count) local;
}

// data const arguments
unsigned data_const_global_count __unsafe_late_const;

struct struct_data_const_global_count {
  int *__counted_by(data_const_global_count) buf;
};

int fun_data_const_global_count(int *__counted_by(data_const_global_count) arg);

void test_local_data_const_global_count() {
  int *__counted_by(data_const_global_count) local;
}

// const function calls
#define __pure2 __attribute__((const))

__pure2 unsigned fun_const_no_argument();

struct struct_consteval_function_call_count {
  int *__counted_by(fun_const_no_argument()) buf;
};

void fun_consteval_function_call_count(int *__counted_by(fun_const_no_argument()) arg);

void test_local_consteval_function_call_count() {
  int *__counted_by(fun_const_no_argument()) local;
}

// expected-warning@+1{{'const' attribute on function returning 'void'; attribute ignored}}
__pure2 void fun_const_void();
struct struct_void_function_call_count {
  // expected-error@+1{{'__counted_by' attribute requires an integer type argument}}
  int *__counted_by(fun_const_void()) buf;
  // expected-error@+1{{'__sized_by' attribute requires an integer type argument}}
  int *__sized_by(fun_const_void()) buf2;
  // expected-error@+1{{'__counted_by_or_null' attribute requires an integer type argument}}
  int *__counted_by_or_null(fun_const_void()) buf3;
  // expected-error@+1{{'__sized_by_or_null' attribute requires an integer type argument}}
  int *__sized_by_or_null(fun_const_void()) buf4;
};

// expected-note@+1 12{{'fun_no_argument' declared here}}
unsigned fun_no_argument();

struct struct_function_call_count {
  // expected-error@+1{{argument of '__counted_by' attribute can only reference function with 'const' attribute}}
  int *__counted_by(fun_no_argument()) buf;
  // expected-error@+1{{argument of '__sized_by' attribute can only reference function with 'const' attribute}}
  int *__sized_by(fun_no_argument()) buf2;
  // expected-error@+1{{argument of '__counted_by_or_null' attribute can only reference function with 'const' attribute}}
  int *__counted_by_or_null(fun_no_argument()) buf3;
  // expected-error@+1{{argument of '__sized_by_or_null' attribute can only reference function with 'const' attribute}}
  int *__sized_by_or_null(fun_no_argument()) buf4;
};

// expected-error@+1{{argument of '__counted_by' attribute can only reference function with 'const' attribute}}
void fun_function_call_count(int *__counted_by(fun_no_argument()) arg);
// expected-error@+1{{argument of '__sized_by' attribute can only reference function with 'const' attribute}}
void fun_function_call_size(int *__sized_by(fun_no_argument()) arg);
// expected-error@+1{{argument of '__counted_by_or_null' attribute can only reference function with 'const' attribute}}
void fun_function_call_count_null(int *__counted_by_or_null(fun_no_argument()) arg);
// expected-error@+1{{argument of '__sized_by_or_null' attribute can only reference function with 'const' attribute}}
void fun_function_call_size_null(int *__sized_by_or_null(fun_no_argument()) arg);

void test_local_function_call_count() {
  // expected-error@+1{{argument of '__counted_by' attribute can only reference function with 'const' attribute}}
  int *__counted_by(fun_no_argument()) local;
  // expected-error@+1{{argument of '__sized_by' attribute can only reference function with 'const' attribute}}
  int *__sized_by(fun_no_argument()) local2;
  // expected-error@+1{{argument of '__counted_by_or_null' attribute can only reference function with 'const' attribute}}
  int *__counted_by_or_null(fun_no_argument()) local3;
  // expected-error@+1{{argument of '__sized_by_or_null' attribute can only reference function with 'const' attribute}}
  int *__sized_by_or_null(fun_no_argument()) local4;
}


// const function calls with argument
__pure2 int fun_const_with_argument(int arg);
__pure2 long fun_const_with_argument2(int arg0, const void *arg1);


struct struct_const_function_argument_count {
  int *__counted_by(fun_const_with_argument(2)) buf;
};

void fun_const_function_argument_count(int *__counted_by(fun_const_with_argument(4)) arg);

void test_local_const_function_argument_count() {
  int *__counted_by(fun_const_with_argument(0)) local;
}

struct struct_const_function_argument_count2 {
  int field;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(field)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument(field)) buf;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(field)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument(field)) buf2;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(field)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument(field)) buf3;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(field)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument(field)) buf4;
  const int const_field;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(const_field)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument(const_field)) buf5;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(const_field)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument(const_field)) buf6;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(const_field)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument(const_field)) buf7;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(const_field)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument(const_field)) buf8;

  int *__counted_by(fun_const_with_argument(const_global_count)) buf9;
  int *__sized_by(fun_const_with_argument(const_global_count)) buf10;
  int *__counted_by_or_null(fun_const_with_argument(const_global_count)) buf11;
  int *__sized_by_or_null(fun_const_with_argument(const_global_count)) buf12;

  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument(data_const_global_count)) buf13;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument(data_const_global_count)) buf14;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument(data_const_global_count)) buf15;
  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument(data_const_global_count)) buf16;
};

// expected-error@+2{{argument of function call 'fun_const_with_argument(global_count)' in '__counted_by' attribute is not a constant expression}}
void fun_const_function_argument_count2(
     int *__counted_by(fun_const_with_argument(global_count)) arg);
// expected-error@+2{{argument of function call 'fun_const_with_argument(global_count)' in '__sized_by' attribute is not a constant expression}}
void fun_const_function_argument_size2(
     int *__sized_by(fun_const_with_argument(global_count)) arg);
// expected-error@+2{{argument of function call 'fun_const_with_argument(global_count)' in '__counted_by_or_null' attribute is not a constant expression}}
void fun_const_function_argument_count_null2(
     int *__counted_by_or_null(fun_const_with_argument(global_count)) arg);
// expected-error@+2{{argument of function call 'fun_const_with_argument(global_count)' in '__sized_by_or_null' attribute is not a constant expression}}
void fun_const_function_argument_size_null2(
     int *__sized_by_or_null(fun_const_with_argument(global_count)) arg);

void fun_const_function_argument_count3(
     int *__counted_by(fun_const_with_argument(const_global_count)) arg);
void fun_const_function_argument_size3(
     int *__sized_by(fun_const_with_argument(const_global_count)) arg);
void fun_const_function_argument_count_null3(
     int *__counted_by_or_null(fun_const_with_argument(const_global_count)) arg);
void fun_const_function_argument_size_null3(
     int *__sized_by_or_null(fun_const_with_argument(const_global_count)) arg);

// expected-error@+2{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__counted_by' attribute is not a constant expression}}
void fun_const_function_argument_count4(
     int *__counted_by(fun_const_with_argument(data_const_global_count)) arg);
// expected-error@+2{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__sized_by' attribute is not a constant expression}}
void fun_const_function_argument_size4(
     int *__sized_by(fun_const_with_argument(data_const_global_count)) arg);
// expected-error@+2{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__counted_by_or_null' attribute is not a constant expression}}
void fun_const_function_argument_count_null4(
     int *__counted_by_or_null(fun_const_with_argument(data_const_global_count)) arg);
// expected-error@+2{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__sized_by_or_null' attribute is not a constant expression}}
void fun_const_function_argument_size_null4(
     int *__sized_by_or_null(fun_const_with_argument(data_const_global_count)) arg);

void test_local_const_function_argument_count2(int arg) {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(arg)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument(arg)) local;
}
void test_local_const_function_argument_size2(int arg) {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(arg)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument(arg)) local;
}
void test_local_const_function_argument_count_null2(int arg) {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(arg)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument(arg)) local;
}
void test_local_const_function_argument_size_null2(int arg) {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(arg)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument(arg)) local;
}

void test_local_const_function_argument_count3() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(global_count)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument(global_count)) local;
}
void test_local_const_function_argument_size3() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(global_count)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument(global_count)) local;
}
void test_local_const_function_argument_count_null3() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(global_count)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument(global_count)) local;
}
void test_local_const_function_argument_size_null3() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(global_count)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument(global_count)) local;
}

void test_local_const_function_argument_count4() {
  int *__counted_by(fun_const_with_argument(const_global_count)) local;
}
void test_local_const_function_argument_size4() {
  int *__sized_by(fun_const_with_argument(const_global_count)) local;
}
void test_local_const_function_argument_count_null4() {
  int *__counted_by_or_null(fun_const_with_argument(const_global_count)) local;
}
void test_local_const_function_argument_size_null4() {
  int *__sized_by_or_null(fun_const_with_argument(const_global_count)) local;
}

void test_local_const_function_argument_count5() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument(data_const_global_count)) local;
}
void test_local_const_function_argument_size5() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument(data_const_global_count)) local;
}
void test_local_const_function_argument_count_null5() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument(data_const_global_count)) local;
}
void test_local_const_function_argument_size_null5() {
  // expected-error@+1{{argument of function call 'fun_const_with_argument(data_const_global_count)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument(data_const_global_count)) local;
}

struct struct_const_function_argument2_count {
  int field0;
  void *field1;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(field0, field1)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument2(field0, field1)) buf;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(field0, field1)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument2(field0, field1)) buf2;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(field0, field1)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument2(field0, field1)) buf3;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(field0, field1)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument2(field0, field1)) buf4;

  const int const_field0;
  const void *const_field1;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, field1)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument2(const_field0, field1)) buf5;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, field1)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument2(const_field0, field1)) buf6;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, field1)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument2(const_field0, field1)) buf7;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, field1)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument2(const_field0, field1)) buf8;

  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, const_field1)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument2(const_field0, const_field1)) buf9;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, const_field1)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument2(const_field0, const_field1)) buf10;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, const_field1)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument2(const_field0, const_field1)) buf11;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_field0, const_field1)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument2(const_field0, const_field1)) buf12;

  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_global_count, const_field1)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument2(const_global_count, const_field1)) buf13;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_global_count, const_field1)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument2(const_global_count, const_field1)) buf14;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_global_count, const_field1)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument2(const_global_count, const_field1)) buf15;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(const_global_count, const_field1)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument2(const_global_count, const_field1)) buf16;

  // expected-error@+1{{argument of function call 'fun_const_with_argument2(data_const_global_count, const_field1)' in '__counted_by' attribute is not a constant expression}}
  int *__counted_by(fun_const_with_argument2(data_const_global_count, const_field1)) buf17;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(data_const_global_count, const_field1)' in '__sized_by' attribute is not a constant expression}}
  int *__sized_by(fun_const_with_argument2(data_const_global_count, const_field1)) buf18;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(data_const_global_count, const_field1)' in '__counted_by_or_null' attribute is not a constant expression}}
  int *__counted_by_or_null(fun_const_with_argument2(data_const_global_count, const_field1)) buf19;
  // expected-error@+1{{argument of function call 'fun_const_with_argument2(data_const_global_count, const_field1)' in '__sized_by_or_null' attribute is not a constant expression}}
  int *__sized_by_or_null(fun_const_with_argument2(data_const_global_count, const_field1)) buf20;
};
