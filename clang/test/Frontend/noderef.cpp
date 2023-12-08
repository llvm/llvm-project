// RUN: %clang_cc1 -fblocks -verify %s

/**
 * Test 'noderef' attribute with c++ constructs.
 */

#define NODEREF __attribute__((noderef))

// Stub out types for 'typeid' to work.
namespace std {
class type_info {};
} // namespace std

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
  int *member4;

  int func() { return member; }
  virtual int func_virt() { return member; }

  A(NODEREF int *x) : member4(x) {} // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
};

class Child : public A {};

void MemberPointer() {
  int NODEREF A::*var = &A::member; // expected-warning{{'noderef' can only be used on an array or pointer type}}
}

int MethodCall(NODEREF A *a) { // expected-note{{a declared here}}
  return a->func();            // expected-warning{{dereferencing a; was declared with a 'noderef' type}}
}

int ChildCall(NODEREF Child *child) { // expected-note{{child declared here}}
  return child->func();               // expected-warning{{dereferencing child; was declared with a 'noderef' type}}
}

std::type_info TypeIdPolymorphic(NODEREF A *a) { // expected-note{{a declared here}}
  return typeid(*a);                             // expected-warning{{dereferencing a; was declared with a 'noderef' type}}
}

class SimpleClass {
  int a;
};

std::type_info TypeIdNonPolymorphic(NODEREF SimpleClass *simple) {
  return typeid(*simple);
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

void cast_from_void_ptr(NODEREF void *x) {
  int *a = static_cast<int *>(x);      // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}

  // Allow regular C-style casts and C-style through reinterpret_casts to be holes
  int *b = reinterpret_cast<int *>(x);
  int *c = (int *)x;
}

void conversion_sequences() {
  NODEREF int *x;
  int *x2 = x;                     // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  int *x3 = static_cast<int *>(x); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
  int *x4 = reinterpret_cast<int *>(x);

  // Functional cast - This is exactly equivalent to a C-style cast.
  typedef int *INT_PTR;
  int *x5 = INT_PTR(x);

  NODEREF Child *child;
  Child *child2 = dynamic_cast<Child *>(child); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

int *static_cast_from_same_ptr_type(NODEREF int *x) {
  return static_cast<int *>(x); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

A *dynamic_cast_up(NODEREF Child *child) {
  return dynamic_cast<A *>(child); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

Child *dynamic_cast_down(NODEREF A *a) {
  return dynamic_cast<Child *>(a); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

A *dynamic_cast_side(NODEREF A *a) {
  return dynamic_cast<A *>(a); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

void *dynamic_cast_to_void_ptr(NODEREF A *a) {
  return dynamic_cast<void *>(a); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

int *const_cast_check(NODEREF const int *x) {
  return const_cast<int *>(x); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}

const int *const_cast_check(NODEREF int *x) {
  return const_cast<const int *>(x); // expected-warning{{casting to dereferenceable pointer removes 'noderef' attribute}}
}
