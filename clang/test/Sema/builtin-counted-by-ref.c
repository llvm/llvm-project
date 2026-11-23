// RUN: %clang_cc1 -std=c99 -fsyntax-only -verify %s

typedef unsigned long int size_t;

int global_array[42];
int global_int;

struct fam_struct {
  int x;
  char count;
  int array[] __attribute__((counted_by(count)));
};

void g(char *);
void h(char);

void test1(struct fam_struct *ptr, int size, int idx) {
  size_t size_of = sizeof(__builtin_counted_by_ref(ptr->array));    // ok
  int align_of = __alignof(__builtin_counted_by_ref(ptr->array));   // ok
  size_t __ignored_assignment;

  *__builtin_counted_by_ref(ptr->array) = size;                     // ok
  *_Generic(__builtin_counted_by_ref(ptr->array),
           void *: &__ignored_assignment,
           default: __builtin_counted_by_ref(ptr->array)) = 42;     // ok
  h(*__builtin_counted_by_ref(ptr->array));                         // ok
}

void test2(struct fam_struct *ptr, int idx) {
  __builtin_counted_by_ref();                                       // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_counted_by_ref(ptr->array, ptr->x, ptr->count);         // expected-error {{too many arguments to function call, expected 1, have 3}}
}

void test3(struct fam_struct *ptr, int idx) {
  __builtin_counted_by_ref(&ptr->array[0]);                         // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(&ptr->array[idx]);                       // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(&ptr->array);                            // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(ptr->x);                                 // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(&ptr->x);                                // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(global_array);                           // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(global_int);                             // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(&global_int);                            // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
}

void test4(struct fam_struct *ptr, int idx) {
  __builtin_counted_by_ref(ptr++->array);                           // expected-error {{'__builtin_counted_by_ref' argument cannot have side-effects}}
  __builtin_counted_by_ref(&ptr->array[idx++]);                     // expected-error {{'__builtin_counted_by_ref' argument cannot have side-effects}}
}

void *test5(struct fam_struct *ptr, int size, int idx) {
  char *ref = __builtin_counted_by_ref(ptr->array);                 // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable}}
  char *int_ptr;
  char *p;

  ref = __builtin_counted_by_ref(ptr->array);                       // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable}}
  g(__builtin_counted_by_ref(ptr->array));                          // expected-error {{value returned by '__builtin_counted_by_ref' cannot be passed into a function}}
  g(ref = __builtin_counted_by_ref(ptr->array));                    // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable}}

  if ((ref = __builtin_counted_by_ref(ptr->array)))                 // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable}}
    ;

  for (p = __builtin_counted_by_ref(ptr->array); p && *p; ++p)      // expected-error {{value returned by '__builtin_counted_by_ref' cannot be assigned to a variable}}
    ;

  return __builtin_counted_by_ref(ptr->array);                      // expected-error {{value returned by '__builtin_counted_by_ref' cannot be returned from a function}}
}

void test6(struct fam_struct *ptr, int size, int idx) {
  *(__builtin_counted_by_ref(ptr->array) + 4) = 37;                 // expected-error {{value returned by '__builtin_counted_by_ref' cannot be used in a binary expression}}
  __builtin_counted_by_ref(ptr->array)[3] = 37;                     // expected-error {{value returned by '__builtin_counted_by_ref' cannot be used in an array subscript expression}}
}

struct non_fam_struct {
  char x;
  long *pointer;
  int array[42];
  short count;
};

void *test7(struct non_fam_struct *ptr, int size) {
  // Arrays and pointers without counted_by return void*
  _Static_assert(_Generic(__builtin_counted_by_ref(ptr->array), void * : 1, default : 0) == 1, "should be void*");
  _Static_assert(_Generic(__builtin_counted_by_ref(ptr->pointer), void * : 1, default : 0) == 1, "should be void*");
  // These are not direct member accesses, so they error
  __builtin_counted_by_ref(&ptr->array[0]);                         // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
  __builtin_counted_by_ref(&ptr->pointer[0]);                       // expected-error {{'__builtin_counted_by_ref' argument must reference a member with the 'counted_by' attribute}}
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

// Tests for pointer members with counted_by attribute
struct ptr_char_count {
  char count;
  int *ptr __attribute__((counted_by(count)));
} *pcp;

struct ptr_short_count {
  short count;
  int *ptr __attribute__((counted_by(count)));
} *psp;

struct ptr_int_count {
  int count;
  int *ptr __attribute__((counted_by(count)));
} *pip;

struct ptr_unsigned_count {
  unsigned count;
  int *ptr __attribute__((counted_by(count)));
} *pup;

struct ptr_long_count {
  long count;
  int *ptr __attribute__((counted_by(count)));
} *plp;

struct ptr_unsigned_long_count {
  unsigned long count;
  int *ptr __attribute__((counted_by(count)));
} *pulp;

void test9(struct ptr_int_count *ptr, int size) {
  size_t size_of = sizeof(__builtin_counted_by_ref(ptr->ptr));      // ok
  int align_of = __alignof(__builtin_counted_by_ref(ptr->ptr));     // ok
  size_t __ignored_assignment;

  *__builtin_counted_by_ref(ptr->ptr) = size;                       // ok
  *_Generic(__builtin_counted_by_ref(ptr->ptr),
           void *: &__ignored_assignment,
           default: __builtin_counted_by_ref(ptr->ptr)) = 42;       // ok
}

void test10(void) {
  _Static_assert(_Generic(__builtin_counted_by_ref(pcp->ptr), char * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(psp->ptr), short * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(pip->ptr), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(pup->ptr), unsigned int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(plp->ptr), long * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(pulp->ptr), unsigned long * : 1, default : 0) == 1, "wrong return type");
}

// Pointer without counted_by returns void*
struct ptr_no_attr {
  int count;
  int *ptr;  // No counted_by attribute
};

void test11(struct ptr_no_attr *p) {
  _Static_assert(_Generic(__builtin_counted_by_ref(p->ptr), void * : 1, default : 0) == 1, "should be void*");
}

struct bitfield_counter {
  int count : 13; // expected-note {{bit-field 'count' declared here}}
  int array[] __attribute__((counted_by(count)));
};

void test_bitfield_counter(struct bitfield_counter *p) {
  __builtin_counted_by_ref(p->array); // expected-error {{'__builtin_counted_by_ref' cannot be used on 'p->array' because its 'counted_by' attribute refers to bit-field 'count'}}
}

// The bit-field restriction also applies to 'counted_by' pointers, not just
// flexible array members, and to a bit-field counter that isn't at the start of
// its storage unit.
struct bitfield_counter_ptr {
  unsigned flags : 3;
  unsigned count : 13; // expected-note {{bit-field 'count' declared here}}
  int *ptr __attribute__((counted_by(count)));
};

void test_bitfield_counter_ptr(struct bitfield_counter_ptr *p) {
  __builtin_counted_by_ref(p->ptr); // expected-error {{'__builtin_counted_by_ref' cannot be used on 'p->ptr' because its 'counted_by' attribute refers to bit-field 'count'}}
}

// Only 'counted_by' is supported. The '_or_null' and 'sized_by' variants return
// 'void *', the same as a member without any attribute, so that the _Generic
// idiom from the documentation still compiles for them.
struct counted_by_or_null_ptr {
  int count;
  int *ptr __attribute__((counted_by_or_null(count)));
};

struct sized_by_ptr {
  int count;
  char *ptr __attribute__((sized_by(count)));
};

struct sized_by_or_null_ptr {
  int count;
  char *ptr __attribute__((sized_by_or_null(count)));
};

void test_or_null_kinds(struct counted_by_or_null_ptr *a, struct sized_by_ptr *b,
                        struct sized_by_or_null_ptr *c) {
  _Static_assert(_Generic(__builtin_counted_by_ref(a->ptr), void * : 1, default : 0) == 1, "should be void*");
  _Static_assert(_Generic(__builtin_counted_by_ref(b->ptr), void * : 1, default : 0) == 1, "should be void*");
  _Static_assert(_Generic(__builtin_counted_by_ref(c->ptr), void * : 1, default : 0) == 1, "should be void*");
}

// The returned pointer's pointee type is the counter's declared type, including
// qualifiers, enums, and typedefs.
enum count_enum { CE_ZERO, CE_ONE };
typedef unsigned short count_t;

struct volatile_count {
  volatile int count;
  int array[] __attribute__((counted_by(count)));
};

struct const_count {
  const int count;
  int array[] __attribute__((counted_by(count)));
};

struct enum_count {
  enum count_enum count;
  int array[] __attribute__((counted_by(count)));
};

struct typedef_count {
  count_t count;
  int array[] __attribute__((counted_by(count)));
};

void test_qualified_enum_typedef(struct volatile_count *vp, struct const_count *cp,
                                 struct enum_count *ep, struct typedef_count *tp) {
  _Static_assert(_Generic(__builtin_counted_by_ref(vp->array), volatile int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(cp->array), const int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(ep->array), enum count_enum * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(tp->array), count_t * : 1, default : 0) == 1, "wrong return type");

  *__builtin_counted_by_ref(vp->array) = 42;                        // ok
  *__builtin_counted_by_ref(cp->array) = 42;                        // expected-error {{read-only variable is not assignable}}
}

// Various ways of naming the flexible array member, including through nested
// and anonymous structs.
struct nested_inner {
  int count;
  int array[] __attribute__((counted_by(count)));
};

struct nested_outer {
  long tag;
  struct nested_inner inner;
};

struct anon_count {
  long tag;
  struct {
    int pad;
    short count;
  };
  int array[] __attribute__((counted_by(count)));
};

void test_nested_anonymous_by_value_subscripted(struct nested_inner *p, struct nested_inner s,
                                                struct nested_outer *o, struct anon_count *a) {
  _Static_assert(_Generic(__builtin_counted_by_ref((*p).array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(p[1].array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref((p)->array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(s.array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(o->inner.array), int * : 1, default : 0) == 1, "wrong return type");
  _Static_assert(_Generic(__builtin_counted_by_ref(a->array), short * : 1, default : 0) == 1, "wrong return type");
}
