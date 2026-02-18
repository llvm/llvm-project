// RUN: %clang_analyze_cc1 -analyzer-checker=core -analyzer-output=text -verify %s

extern char *something();

void test1() {
  int *p;
  p = (int *)22; // expected-note{{Pointer value of (int *)22 stored to 'p'}}
  *p = 2; // expected-warning{{Dereference of a fixed address (loaded from variable 'p')}} \
          // expected-note{{Dereference of a fixed address (loaded from variable 'p')}}
}

void test2_1(int *p) {
  *p = 1; // expected-warning{{Dereference of a fixed address (loaded from variable 'p')}} \
          // expected-note{{Dereference of a fixed address (loaded from variable 'p')}}
}

void test2() {
  int *p = (int *)11; // expected-note{{'p' initialized to (int *)11}}
  test2_1(p); // expected-note{{Passing pointer value (int *)11 via 1st parameter 'p'}} \
              // expected-note{{Calling 'test2_1'}}
}

struct test3_s {
  int a;
};


void test3() {
  struct test3_s *x;
  unsigned long long val = 1111111; // expected-note{{'val' initialized to 1111111}}
  x = (struct test3_s *)val; // expected-note{{Pointer value of (struct test3_s *)1111111 stored to 'x'}}
  x->a = 3; // expected-warning{{Access to field 'a' results in a dereference of a fixed address (loaded from variable 'x')}} \
            // expected-note{{Access to field 'a' results in a dereference of a fixed address (loaded from variable 'x')}}
}

char *test4_1() {
  char *ret;
  ret = something(); // expected-note{{Value assigned to 'ret'}}
  if (ret == (char *)-1) // expected-note{{Assuming the condition is true}} \
                         // expected-note{{Taking true branch}}
    return ret; // expected-note{{Returning pointer (loaded from 'ret')}}
  return 0;
}

void test4() {
  char *x;
  x = test4_1(); // expected-note{{Calling 'test4_1'}} \
                 // expected-note{{Returning from 'test4_1'}} \
                 // expected-note{{Pointer value of (char *)-1 stored to 'x'}}
  *x = 3; // expected-warning{{Dereference of a fixed address (loaded from variable 'x')}} \
          // expected-note{{Dereference of a fixed address (loaded from variable 'x')}}
}

void suppress_volatile_pointee(void) {
  *(volatile int *)0x00011100 = 4; // no-warning: volatile pointees are suppressed
}

void suppress_volatile_pointee_using_subscript(void) {
  ((volatile int *)0x00011100)[0] = 4; // no-warning: volatile pointees are suppressed
}

void suppress_ptr_to_100element_volatile_array(void) {
  ((volatile int (*)[100])0x00011100)[2][0] = 4; // no-warning: volatile pointees are suppressed
}

void deref_volatile_nullptr(void) {
  *(volatile int *)0 = 1; // core.NullDereference still warns about this
  // expected-warning@-1 {{Dereference of null pointer [core.NullDereference]}}
  // expected-note@-2    {{Dereference of null pointer}}
}
