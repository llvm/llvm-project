// RUN: %clang_cc1 -verify -std=c++11 %s

int foo() {
  int x[2]; // expected-note 4 {{array 'x' declared here}}
  int y[2]; // expected-note 2 {{array 'y' declared here}}
  int z[1]; // expected-note {{array 'z' declared here}}
  int w[1][1]; // expected-note {{array 'w' declared here}}
  int v[1][1][1]; // expected-note {{array 'v' declared here}}
  int *p = &y[2]; // no-warning
  (void) sizeof(x[2]); // no-warning
  y[2] = 2; // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
  z[1] = 'x'; // expected-warning {{array index 1 is past the end of the array (which contains 1 element)}}
  w[0][2] = 0; // expected-warning {{array index 2 is past the end of the array (which contains 1 element)}}
  v[0][0][2] = 0; // expected-warning {{array index 2 is past the end of the array (which contains 1 element)}}
  return x[2] +  // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
         y[-1] + // expected-warning {{array index -1 is before the beginning of the array}}
         x[sizeof(x)] +  // expected-warning {{array index 8 is past the end of the array (which contains 2 elements)}}
         x[sizeof(x) / sizeof(x[0])] +  // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
         x[sizeof(x) / sizeof(x[0]) - 1] + // no-warning
         x[sizeof(x[2])]; // expected-warning {{array index 4 is past the end of the array (which contains 2 elements)}}
}

// This code example tests that -Warray-bounds works with arrays that
// are template parameters.
template <char *sz> class Qux {
  bool test() { return sz[0] == 'a'; }
};

void f1(int a[1]) {
  int val = a[3]; // no warning for function argumnet
}

void f2(const int (&a)[2]) { // expected-note {{declared here}}
  int val = a[3];  // expected-warning {{array index 3 is past the end of the array (which contains 2 elements)}}
}

void test() {
  struct {
    int a[0];
  } s2;
  s2.a[3] = 0; // no warning for 0-sized array

  union {
    short a[2]; // expected-note 4 {{declared here}}
    char c[4];
  } u;
  u.a[3] = 1; // expected-warning {{array index 3 is past the end of the array (which contains 2 elements)}}
  u.c[3] = 1; // no warning
  short *p = &u.a[2]; // no warning
  p = &u.a[3]; // expected-warning {{array index 3 is past the end of the array (which contains 2 elements)}}
  *(&u.a[2]) = 1; // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
  *(&u.a[3]) = 1; // expected-warning {{array index 3 is past the end of the array (which contains 2 elements)}}
  *(&u.c[3]) = 1; // no warning

  const int const_subscript = 3;
  int array[2]; // expected-note {{declared here}}
  array[const_subscript] = 0;  // expected-warning {{array index 3 is past the end of the array (which contains 2 elements)}}

  int *ptr;
  ptr[3] = 0; // no warning for pointer references
  int array2[] = { 0, 1, 2 }; // expected-note 2 {{declared here}}

  array2[3] = 0; // expected-warning {{array index 3 is past the end of the array (which contains 3 elements)}}
  array2[2+2] = 0; // expected-warning {{array index 4 is past the end of the array (which contains 3 elements)}}

  const char *str1 = "foo";
  char c1 = str1[5]; // no warning for pointers

  const char str2[] = "foo"; // expected-note {{declared here}}
  char c2 = str2[5]; // expected-warning {{array index 5 is past the end of the array (which contains 4 elements)}}

  int (*array_ptr)[2];
  (*array_ptr)[3] = 1; // expected-warning {{array index 3 is past the end of the array (which contains 2 elements)}}
}

template <int I> struct S {
  char arr[I]; // expected-note 3 {{declared here}}
};
template <int I> void f() {
  S<3> s;
  s.arr[4] = 0; // expected-warning 2 {{array index 4 is past the end of the array (which contains 3 elements)}}
  s.arr[I] = 0; // expected-warning {{array index 5 is past the end of the array (which contains 3 elements)}}
}

void test_templates() {
  f<5>(); // expected-note {{in instantiation}}
}

#define SIZE 10
#define ARR_IN_MACRO(flag, arr, idx) flag ? arr[idx] : 1

int test_no_warn_macro_unreachable() {
  int arr[SIZE]; // expected-note {{array 'arr' declared here}}
  return ARR_IN_MACRO(0, arr, SIZE) + // no-warning
         ARR_IN_MACRO(1, arr, SIZE); // expected-warning{{array index 10 is past the end of the array (which contains 10 elements)}}
}

// This exhibited an assertion failure for a 32-bit build of Clang.
int test_pr9240() {
  short array[100]; // expected-note {{array 'array' declared here}}
  return array[(unsigned long long) 100]; // expected-warning {{array index 100 is past the end of the array (which contains 100 elements)}}
}

// PR 9284 - a template parameter can cause an array bounds access to be
// infeasible.
template <bool extendArray>
void pr9284() {
    int arr[3 + (extendArray ? 1 : 0)];

    if (extendArray)
        arr[3] = 42; // no-warning
}

template <bool extendArray>
void pr9284b() {
    int arr[3 + (extendArray ? 1 : 0)]; // expected-note {{array 'arr' declared here}}

    if (!extendArray)
        arr[3] = 42; // expected-warning{{array index 3 is past the end of the array (which contains 3 elements)}}
}

void test_pr9284() {
    pr9284<true>();
    pr9284<false>();
    pr9284b<true>();
    pr9284b<false>(); // expected-note{{in instantiation of function template specialization 'pr9284b<false>' requested here}}
}

int test_pr9296() {
    int array[2];
    return array[true]; // no-warning
}

int test_sizeof_as_condition(int flag) {
  int arr[2] = { 0, 0 }; // expected-note {{array 'arr' declared here}}
  if (flag) 
    return sizeof(char) != sizeof(char) ? arr[2] : arr[1];
  return sizeof(char) == sizeof(char) ? arr[2] : arr[1]; // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
}

void test_switch() {
  switch (4) {
    case 1: {
      int arr[2];
      arr[2] = 1; // no-warning
      break;
    }
    case 4: {
      int arr[2]; // expected-note {{array 'arr' declared here}}
      arr[2] = 1; // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
      break;
    }
    default: {
      int arr[2];
      arr[2] = 1; // no-warning
      break;
    }
  }
}

// Test nested switch statements.
enum enumA { enumA_A, enumA_B, enumA_C, enumA_D, enumA_E };
enum enumB { enumB_X, enumB_Y, enumB_Z };
static enum enumB myVal = enumB_X;
void test_nested_switch() {
  switch (enumA_E) { // expected-warning {{no case matching constant}}
    switch (myVal) { // expected-warning {{enumeration values 'enumB_X' and 'enumB_Z' not handled in switch}} expected-note {{add missing switch cases}}
      case enumB_Y: ;
    }
  }
}

// Test that if all the values of an enum covered, that the 'default' branch
// is unreachable.
enum Values { A, B, C, D };
void test_all_enums_covered(enum Values v) {
  int x[2];
  switch (v) {
  case A: return;
  case B: return;
  case C: return;
  case D: return;
  }
  x[2] = 0; // no-warning
}

namespace tailpad {
  struct foo {
    char c1[1]; // expected-note {{declared here}}
    int x;
    char c2[1];
  };

  class baz {
   public:
    char c1[1]; // expected-note {{declared here}}
    int x;
    char c2[1];
  };

  char bar(struct foo *F, baz *B) {
    return F->c1[3] + // expected-warning {{array index 3 is past the end of the array (which contains 1 element)}}
           F->c2[3] + // no warning, foo could have tail padding allocated.
           B->c1[3] + // expected-warning {{array index 3 is past the end of the array (which contains 1 element)}}
           B->c2[3]; // no warning, baz could have tail padding allocated.
  }
}

namespace metaprogramming {
#define ONE 1
  struct foo { char c[ONE]; }; // expected-note {{declared here}}
  template <int N> struct bar { char c[N]; }; // expected-note {{declared here}}

  char test(foo *F, bar<1> *B) {
    return F->c[3] + // expected-warning {{array index 3 is past the end of the array (which contains 1 element)}}
           B->c[3]; // expected-warning {{array index 3 is past the end of the array (which contains 1 element)}}
  }
}

void bar(int x) {}
int test_more() {
  int foo[5]; // expected-note 5 {{array 'foo' declared here}}
  bar(foo[5]); // expected-warning {{array index 5 is past the end of the array (which contains 5 elements)}}
  ++foo[5]; // expected-warning {{array index 5 is past the end of the array (which contains 5 elements)}}
  if (foo[6]) // expected-warning {{array index 6 is past the end of the array (which contains 5 elements)}}
    return --foo[6]; // expected-warning {{array index 6 is past the end of the array (which contains 5 elements)}}
  else
    return foo[5]; // expected-warning {{array index 5 is past the end of the array (which contains 5 elements)}}
}

void test_pr10771() {
    double foo[4096];  // expected-note {{array 'foo' declared here}}

    ((char*)foo)[sizeof(foo) - 1] = '\0';  // no-warning
    *(((char*)foo) + sizeof(foo) - 1) = '\0';  // no-warning

    ((char*)foo)[sizeof(foo)] = '\0';  // expected-warning {{array index 32768 is past the end of the array (which contains 32768 elements)}}

    // TODO: This should probably warn, too.
    *(((char*)foo) + sizeof(foo)) = '\0';  // no-warning
}

int test_pr11007_aux(const char * restrict, ...);
  
// Test checking with varargs.
void test_pr11007() {
  double a[5]; // expected-note {{array 'a' declared here}}
  test_pr11007_aux("foo", a[1000]); // expected-warning {{array index 1000 is past the end of the array}}
}

void test_rdar10916006(void)
{
	int a[128]; // expected-note {{array 'a' declared here}}
	a[(unsigned char)'\xA1'] = 1; // expected-warning {{array index 161 is past the end of the array}}
}

struct P {
  int a;
  int b;
};

void test_struct_array_index() {
  struct P p[10]; // expected-note {{array 'p' declared here}}
  p[11] = {0, 1}; // expected-warning {{array index 11 is past the end of the array (which contains 10 elements)}}
}

int operator+(const struct P &s1, const struct P &s2);
int test_operator_overload_struct_array_index() {
  struct P x[10] = {0}; // expected-note {{array 'x' declared here}}
  return x[1] + x[11]; // expected-warning {{array index 11 is past the end of the array (which contains 10 elements)}}
}

int multi[2][2][2]; // expected-note 3 {{array 'multi' declared here}}
int test_multiarray() {
  return multi[2][0][0] + // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
         multi[0][2][0] + // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
         multi[0][0][2];  // expected-warning {{array index 2 is past the end of the array (which contains 2 elements)}}
}

struct multi_s {
  int arr[4];
};
struct multi_s multi2[4]; // expected-note {{array 'multi2' declared here}}
int test_struct_multiarray() {
  return multi2[4].arr[0]; // expected-warning {{array index 4 is past the end of the array (which contains 4 elements)}}
}

namespace PR39746 {
  struct S;
  extern S xxx[2]; // expected-note {{array 'xxx' declared here}}
  class C {};

  C &f() { return reinterpret_cast<C *>(xxx)[1]; } // no-warning
  // We have no info on whether this is out-of-bounds.
  C &g() { return reinterpret_cast<C *>(xxx)[2]; } // no-warning
  // We can still diagnose this.
  C &h() { return reinterpret_cast<C *>(xxx)[-1]; } // expected-warning {{array index -1 is before the beginning of the array}}
}
