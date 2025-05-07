
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -verify %s
// RUN: %clang_cc1 -fsyntax-only -fbounds-safety -x objective-c -fexperimental-bounds-safety-objc -verify %s
#include <ptrcheck.h>

// data const arguments
unsigned data_const_global_count __unsafe_late_const;
unsigned global_count;
const unsigned const_global_count;

enum { enum_count = 10, };

struct struct_enum_count {
  int *__counted_by(enum_count) buf;
};

void test_enum_count(struct struct_enum_count *sp, int *__bidi_indexable arg) {
  sp->buf = arg;
}

struct struct_data_const_global_count {
  int *__counted_by(data_const_global_count) buf;
};

struct struct_global_count {
  // expected-error@+1{{count expression on struct field may only reference other fields of the same struct}}
  int *__counted_by(global_count) buf;
};

struct struct_const_global_count {
  int *__counted_by(const_global_count) buf;
};

unsigned data_const_global_count_flex __unsafe_late_const;

struct struct_data_const_global_count_flex {
  int dummy;
  int flex[__counted_by(data_const_global_count_flex)];
};

struct struct_global_count_flex {
  int dummy;
  // expected-error@+1{{count expression on struct field may only reference other fields of the same struct}}
  int flex[__counted_by(global_count)];
};
struct struct_const_global_count_flex {
  int dummy;
  int flex[__counted_by(const_global_count)];
};

void fun_update_data_const_global_count_flex(unsigned count) {
  data_const_global_count_flex = 100;
}

void fun_const_global_count(int *__counted_by(data_const_global_count) arg) {
  int arr[10];
  arg = arr;
}

// expected-error@+1{{count expression in function declaration may only reference parameters of that function}}
void fun_global_count(int *__counted_by(global_count) arg);

// expected-error@+1{{count expression in function declaration may only reference parameters of that function}}
void fun_global_array_count(int arg[__counted_by(global_count)]);

void fun_data_const_init() {
  data_const_global_count = 10;
}

void test_local_data_const_global_count(int *__bidi_indexable arg) {
  int *__counted_by(data_const_global_count) local;
  local = arg;
}

void test_local_global_count(int *__bidi_indexable arg) {
  // expected-error@+2{{argument of '__counted_by' attribute cannot refer to declaration from a different scope}}
  // expected-error@+1{{argument of '__counted_by' attribute cannot refer to declaration of a different lifetime}}
  int *__counted_by(global_count) local;
}

void test_data_const_struct_init() {
  struct struct_data_const_global_count s; // better be an error
}

void test_data_const_struct_init2(int *__indexable arg) {
  struct struct_data_const_global_count s = { arg };
}

void test_data_const_struct_init3(int *__indexable arg) {
  struct struct_data_const_global_count s; // better be an error
  s.buf = arg;
}

void test_data_const_struct_init4(void *__bidi_indexable arg) {
  struct struct_data_const_global_count *sp =
        (struct struct_data_const_global_count *)arg;
}

void test_global_const_struct_init(struct struct_const_global_count *sp,
                                   int *__bidi_indexable arg) {
  sp->buf = arg;
}
