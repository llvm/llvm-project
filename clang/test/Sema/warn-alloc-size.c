// RUN: %clang_cc1 -fsyntax-only -verify -Walloc-size %s
struct Foo { int x[10]; };

void *malloc(unsigned long) __attribute__((alloc_size(1)));
void *alloca(unsigned long) __attribute__((alloc_size(1)));
void *calloc(unsigned long, unsigned long) __attribute__((alloc_size(2, 1)));

void foo_consumer(struct Foo* p);

void alloc_foo(void) {
  struct Foo *ptr1 = malloc(sizeof(struct Foo));
  struct Foo *ptr2 = malloc(sizeof(*ptr2));
  struct Foo *ptr3 = calloc(1, sizeof(*ptr3));
  struct Foo *ptr4 = calloc(sizeof(*ptr4), 1);
  struct Foo (*ptr5)[5] = malloc(sizeof(*ptr5));
  void *ptr6 = malloc(4);

  // Test insufficient size with different allocation functions.
  struct Foo *ptr7 = malloc(sizeof(ptr7));      // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr8 = alloca(sizeof(ptr8));      // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr9 = calloc(1, sizeof(ptr9));   // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr10 = calloc(sizeof(ptr10), 1); // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}

  // Test function arguments.
  foo_consumer(malloc(4)); // expected-warning {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}

  // Test explicit cast.
  struct Foo *ptr11 = (struct Foo *)malloc(sizeof(*ptr11));
  struct Foo *ptr12 = (struct Foo *)malloc(sizeof(ptr12));    // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr13 = (struct Foo *)alloca(sizeof(ptr13));    // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr14 = (struct Foo *)calloc(1, sizeof(ptr14)); // expected-warning {{allocation of insufficient size '8' for type 'struct Foo' with size '40'}}
  struct Foo *ptr15 = (struct Foo *)malloc(4);                // expected-warning {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}
  void *ptr16 = (struct Foo *)malloc(4);                      // expected-warning {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}

  struct Foo *ptr17 = (void *)(struct Foo *)malloc(4); // expected-warning 2 {{allocation of insufficient size '4' for type 'struct Foo' with size '40'}}
  int *ptr18 = (unsigned *)(void *)(int *)malloc(1);   // expected-warning {{initializing 'int *' with an expression of type 'unsigned int *' converts between pointers to integer types with different sign}}
                                                       // expected-warning@-1 {{allocation of insufficient size '1' for type 'int' with size '4'}}
                                                       // expected-warning@-2 {{allocation of insufficient size '1' for type 'unsigned int' with size '4'}}
  int *ptr19 = (void *)(int *)malloc(1);               // expected-warning {{allocation of insufficient size '1' for type 'int' with size '4'}}
                                                       // expected-warning@-1 {{allocation of insufficient size '1' for type 'int' with size '4'}}
  (void)(int *)malloc(1);                              // expected-warning {{allocation of insufficient size '1' for type 'int' with size '4'}}
}
