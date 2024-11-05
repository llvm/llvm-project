// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify %s

typedef unsigned long int size_t;

int global_array[42];
int global_int;

struct fam_struct {
  int x;
  char count;
  int array[] __attribute__((counted_by(count)));
};

void test1(struct fam_struct *ptr, int size, int idx) {
  size_t size_of = sizeof(__builtin_counted_by_ref(ptr->array)); // ok

  *__builtin_counted_by_ref(ptr->array) = size;             // ok
  *__builtin_counted_by_ref(&ptr->array[idx]) = size;       // ok
  *__builtin_counted_by_ref(&ptr->array) = size;            // ok


  {
      size_t __ignored_assignment;
      *_Generic(__builtin_counted_by_ref(ptr->array),
               void *: &__ignored_assignment,
               default: __builtin_counted_by_ref(ptr->array)) = 42; // ok
  }
}

void test2(struct fam_struct *ptr, int idx) {
  __builtin_counted_by_ref();                               // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_counted_by_ref(ptr->array, ptr->x, ptr->count); // expected-error {{too many arguments to function call, expected 1, have 3}}
}

void test3(struct fam_struct *ptr, int idx) {
  __builtin_counted_by_ref(ptr->x);                         // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  __builtin_counted_by_ref(&ptr->x);                        // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  __builtin_counted_by_ref(global_array);                   // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  __builtin_counted_by_ref(global_int);                     // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  __builtin_counted_by_ref(&global_int);                    // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
}

void test4(struct fam_struct *ptr, int idx) {
  __builtin_counted_by_ref(ptr++->array);                   // expected-error {{'__builtin_counted_by_ref' argument cannot have side-effects}}
  __builtin_counted_by_ref(&ptr->array[idx++]);             // expected-error {{'__builtin_counted_by_ref' argument cannot have side-effects}}
}

void foo(char *);

void *test5(struct fam_struct *ptr, int size, int idx) {
  char *ref = __builtin_counted_by_ref(ptr->array);         // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}

  ref = __builtin_counted_by_ref(ptr->array);               // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}
  ref = (char *)(int *)(42 + &*__builtin_counted_by_ref(ptr->array)); // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}
  foo(__builtin_counted_by_ref(ptr->array));                // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}
  foo(ref = __builtin_counted_by_ref(ptr->array));          // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}

  if ((ref = __builtin_counted_by_ref(ptr->array)))         // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}
    ;

  for (char *p = __builtin_counted_by_ref(ptr->array); p && *p; ++p) // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}
    ;

  return __builtin_counted_by_ref(ptr->array);              // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable, have its address taken, or passed into or returned from a function}}
}

void test6(struct fam_struct *ptr, int size, int idx) {
  *(__builtin_counted_by_ref(ptr->array) + 4) = 37;         // expected-error {{value returned by '__builtin_counted_by_ref' cannot be used in a binary expression}}
  __builtin_counted_by_ref(ptr->array)[3] = 37;             // expected-error {{value returned by '__builtin_counted_by_ref' cannot be used in an array subscript expression}}
}

struct non_fam_struct {
  char x;
  long *pointer;
  int array[42];
  short count;
};

void *test7(struct non_fam_struct *ptr, int size) {
  *__builtin_counted_by_ref(ptr->array) = size          // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  *__builtin_counted_by_ref(&ptr->array[0]) = size;     // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  *__builtin_counted_by_ref(ptr->pointer) = size;       // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
  *__builtin_counted_by_ref(&ptr->pointer[0]) = size;   // expected-error {{'__builtin_counted_by_ref' argument must reference a flexible array member}}
}

struct char_count {
  char count;
  int array[] __attribute__((counted_by(count)));
} *cp;

struct short_count {
  short count;
  int array[] __attribute__((counted_by(count)));
} *sp;

struct int_count {
  int count;
  int array[] __attribute__((counted_by(count)));
} *ip;

struct unsigned_count {
  unsigned count;
  int array[] __attribute__((counted_by(count)));
} *up;

struct long_count {
  long count;
  int array[] __attribute__((counted_by(count)));
} *lp;

struct unsigned_long_count {
  unsigned long count;
  int array[] __attribute__((counted_by(count)));
} *ulp;

void test8(void) {
  _Static_assert(_Generic(__builtin_counted_by_ref(cp->array), char * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(sp->array), short * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(ip->array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(up->array), unsigned int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(lp->array), long * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(ulp->array), unsigned long * : 1, default : 0) == 1, "wrong return type");
}
