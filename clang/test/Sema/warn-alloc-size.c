// RUN: %clang_cc1 -triple x86_64-linux -fsyntax-only -verify -Walloc-size %s
struct Foo { int x[10]; };

typedef __typeof__(sizeof(int)) size_t;
void *my_malloc(size_t) __attribute__((alloc_size(1)));
void *my_calloc(size_t, size_t) __attribute__((alloc_size(2, 1)));

void foo_consumer(struct Foo* p);

void alloc_foo(void) {
  struct Foo *ptr1 = my_malloc(sizeof(struct Foo));
  struct Foo *ptr2 = my_malloc(sizeof(*ptr2));
  struct Foo *ptr3 = my_calloc(1, sizeof(*ptr3));
  struct Foo *ptr4 = my_calloc(sizeof(*ptr4), 1);
  struct Foo (*ptr5)[5] = my_malloc(sizeof(*ptr5));
  void *ptr6 = my_malloc(4);

  // Test insufficient size with different allocation functions.
  struct Foo *ptr7 = my_malloc(sizeof(ptr7));      // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr8 = my_calloc(1, sizeof(ptr8));   // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr9 = my_calloc(sizeof(ptr9), 1);   // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}

  // Test function arguments.
  foo_consumer(my_malloc(4)); // expected-warning {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}

  // Test explicit cast.
  struct Foo *ptr10 = (struct Foo *)my_malloc(sizeof(*ptr10));
  struct Foo *ptr11 = (struct Foo *)my_malloc(sizeof(ptr11));    // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr12 = (struct Foo *)my_calloc(1, sizeof(ptr12)); // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr13 = (struct Foo *)my_malloc(4);                // expected-warning {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}
  void *ptr14 = (struct Foo *)my_malloc(4);                      // expected-warning {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}

  struct Foo *ptr15 = (void *)(struct Foo *)my_malloc(4); // expected-warning 2 {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}
  int *ptr16 = (unsigned *)(void *)(int *)my_malloc(1);   // expected-warning {{initializing 'int *' with an expression of type 'unsigned int *' converts between pointers to integer types with different sign}}
                                                          // expected-warning@-1 {{allocation of insufficient size '1' for type 'int' with size '4'}}
                                                          // expected-warning@-2 {{allocation of insufficient size '1' for type 'unsigned int' with size '4'}}
  int *ptr17 = (void *)(int *)my_malloc(1);               // expected-warning {{allocation of insufficient size '1' for type 'int' with size '4'}}
                                                          // expected-warning@-1 {{allocation of insufficient size '1' for type 'int' with size '4'}}
  (void)(int *)my_malloc(1);                              // expected-warning {{allocation of insufficient size '1' for type 'int' with size '4'}}

  void *funcptr_1 = (void (*)(int))my_malloc(1);

  // Zero size allocations are assumed to be intentional.
  int *zero_alloc1 = my_malloc(0);
  int *zero_alloc2 = (int *)my_malloc(0);
}
