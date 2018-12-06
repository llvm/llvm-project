// RUN: %clang_cc1 -fblocks -verify %s

/**
 * Test 'noderef' attribute with c++ constructs.
 */

#define NODEREF __attribute__((noderef))

void Normal() {
  int NODEREF i;        // expected-warning{{'noderef' can only be used on an array or pointer type}}
  int NODEREF *i_ptr;   // expected-note 2 {{i_ptr declared here}}
  int NODEREF **i_ptr2; // ok
  int *NODEREF i_ptr3;  // expected-warning{{'noderef' can only be used on an array or pointer type}}
  int *NODEREF *i_ptr4; // ok

  auto NODEREF *auto_i_ptr = i_ptr;
  auto NODEREF auto_i = i; // expected-warning{{'noderef' can only be used on an array or pointer type}}

  struct {
    int x;
    int y;
  } NODEREF *s;

  int __attribute__((noderef(10))) * no_args; // expected-error{{'noderef' attribute takes no arguments}}

  int i2 = *i_ptr;     // expected-warning{{dereferencing i_ptr; was declared with a 'noderef' type}}
  int &i3 = *i_ptr;    // expected-warning{{dereferencing i_ptr; was declared with a 'noderef' type}}
  int *i_ptr5 = i_ptr; // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  int *i_ptr6(i_ptr);  // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

const int NODEREF *const_i_ptr;
static int NODEREF *static_i_ptr;

void ParenTypes() {
  int NODEREF(*i_ptr);    // ok (same as `int NODEREF *`)
  int NODEREF *(*i_ptr2); // ok (same as `int NODEREF **`)
}

// Function declarations
int NODEREF func();   // expected-warning{{'noderef' can only be used on an array or pointer type}}
int NODEREF *func2(); // ok (returning pointer)

typedef int NODEREF (*func3)(int); // expected-warning{{'noderef' can only be used on an array or pointer type}}
typedef int NODEREF *(*func4)(int);

void Arrays() {
  int NODEREF i_arr[10];      // ok
  int NODEREF i_arr2[10][10]; // ok
  int NODEREF *i_arr3[10];    // ok
  int NODEREF i_arr4[] = {1, 2};
}

void ParenArrays() {
  int NODEREF(i_ptr[10]);
  int NODEREF(i_ptr2[10])[10];
}

typedef int NODEREF *(*func5[10])(int);

// Arguments
void func6(int NODEREF x); // expected-warning{{'noderef' can only be used on an array or pointer type}}
void func7(int NODEREF *x);
void func8() NODEREF;

void References() {
  int x = 2;
  int NODEREF &y = x; // expected-warning{{'noderef' can only be used on an array or pointer type}}
  int *xp = &x;
  int NODEREF *&a = xp; // ok (reference to a NODEREF *)
  int *NODEREF &b = xp; // expected-warning{{'noderef' can only be used on an array or pointer type}}
}

void BlockPointers() {
  typedef int NODEREF (^IntBlock)(); // expected-warning{{'noderef' can only be used on an array or pointer type}}
}

class A {
public:
  int member;
  int NODEREF *member2;
  int NODEREF member3; // expected-warning{{'noderef' can only be used on an array or pointer type}}
};

void MemberPointer() {
  int NODEREF A::*var = &A::member; // expected-warning{{'noderef' can only be used on an array or pointer type}}
}

template <class Ty>
class B {
  Ty NODEREF *member;
  Ty NODEREF member2; // expected-warning{{'noderef' can only be used on an array or pointer type}}
};

void test_lambdas() {
  auto l = [](int NODEREF *x){  // expected-note{{x declared here}}
    return *x;  // expected-warning{{dereferencing x; was declared with a 'noderef' type}}
  };
}

int NODEREF *glob_ptr;  // expected-note{{glob_ptr declared here}}
int glob_int = *glob_ptr;  // expected-warning{{dereferencing glob_ptr; was declared with a 'noderef' type}}
