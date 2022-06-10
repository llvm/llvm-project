// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9   -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -fblocks %s -fexceptions -fcxx-exceptions -Wno-tautological-undefined-compare
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin9 -analyzer-checker=core,alpha.core,debug.ExprInspection -verify -fblocks %s -fexceptions -fcxx-exceptions -Wno-tautological-undefined-compare

void clang_analyzer_warnIfReached();

// Test basic handling of references.
char &test1_aux();
char *test1() {
  return &test1_aux();
}

// Test test1_aux() evaluates to char &.
char test1_as_rvalue() {
  return test1_aux();
}

// Test passing a value as a reference.  The 'const' in test2_aux() adds
// an ImplicitCastExpr, which is evaluated as an lvalue.
int test2_aux(const int &n);
int test2(int n) {
  return test2_aux(n);
}

int test2_b_aux(const short &n);
int test2_b(int n) {
  return test2_b_aux(n);
}

// Test getting the lvalue of a derived and converting it to a base.  This
// previously crashed.
class Test3_Base {};
class Test3_Derived : public Test3_Base {};

int test3_aux(Test3_Base &x);
int test3(Test3_Derived x) {
  return test3_aux(x);
}

//===---------------------------------------------------------------------===//
// Test CFG support for C++ condition variables.
//===---------------------------------------------------------------------===//

int test_init_in_condition_aux();
int test_init_in_condition() {
  if (int x = test_init_in_condition_aux()) { // no-warning
    return 1;
  }
  return 0;
}

int test_init_in_condition_switch() {
  switch (int x = test_init_in_condition_aux()) { // no-warning
    case 1:
      return 0;
    case 2:
      if (x == 2)
        return 0;
      else {
        clang_analyzer_warnIfReached();  // unreachable
      }
    default:
      break;
  }
  return 0;
}

int test_init_in_condition_while() {
  int z = 0;
  while (int x = ++z) { // no-warning
    if (x == 2)
      break;
  }
  
  if (z == 2)
    return 0;
  
  clang_analyzer_warnIfReached();  // unreachable
  return 0;
}


int test_init_in_condition_for() {
  int z = 0;
  for (int x = 0; int y = ++z; ++x) {
    if (x == y) // no-warning
      break;
  }
  if (z == 1)
    return 0;
    
  clang_analyzer_warnIfReached();  // unreachable
  return 0;
}

//===---------------------------------------------------------------------===//
// Test handling of 'this' pointer.
//===---------------------------------------------------------------------===//

class TestHandleThis {
  int x;

  TestHandleThis();  
  int foo();
  int null_deref_negative();
  int null_deref_positive();  
};

int TestHandleThis::foo() {
  // Assume that 'x' is initialized.
  return x + 1; // no-warning
}

int TestHandleThis::null_deref_negative() {
  x = 10;
  if (x == 10) {
    return 1;
  }
  clang_analyzer_warnIfReached();  // unreachable
  return 0;  
}

int TestHandleThis::null_deref_positive() {
  x = 10;
  if (x == 9) {
    return 1;
  }
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
  return 0;  
}

// PR 7675 - passing literals by-reference
void pr7675(const double &a);
void pr7675(const int &a);
void pr7675(const char &a);
void pr7675_i(const _Complex double &a);

void pr7675_test() {
  pr7675(10.0);
  pr7675(10);
  pr7675('c');
  pr7675_i(4.0j);

  // Add check to ensure we are analyzing the code up to this point.
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}

// <rdar://problem/8375510> - CFGBuilder should handle temporaries.
struct R8375510 {
  R8375510();
  ~R8375510();
  R8375510 operator++(int);
};

int r8375510(R8375510 x, R8375510 y) {
  for (; ; x++) { }
}

// PR8419 -- this used to crash.

class String8419 {
 public:
  char& get(int n);
  char& operator[](int n);
};

char& get8419();

void Test8419() {
  String8419 s;
  ++(s.get(0));
  get8419()--;  // used to crash
  --s[0];       // used to crash
  s[0] &= 1;    // used to crash
  s[0]++;       // used to crash
}

// PR8426 -- this used to crash.

void Use(void* to);

template <class T> class Foo {
  ~Foo();
  struct Bar;
  Bar* bar_;
};

template <class T> Foo<T>::~Foo() {
  Use(bar_);
  T::DoSomething();
  bar_->Work();
}

// PR8427 -- this used to crash.

class Dummy {};

bool operator==(Dummy, int);

template <typename T>
class Foo2 {
  bool Bar();
};

template <typename T>
bool Foo2<T>::Bar() {
  return 0 == T();
}

// PR8433 -- this used to crash.

template <typename T>
class Foo3 {
 public:
  void Bar();
  void Baz();
  T value_;
};

template <typename T>
void Foo3<T>::Bar() {
  Baz();
  value_();
}

//===---------------------------------------------------------------------===//
// Handle misc. C++ constructs.
//===---------------------------------------------------------------------===//

namespace fum {
  int i = 3;
};

void test_namespace() {
  // Previously triggered a crash.
  using namespace fum;
  int x = i;
}

// Test handling methods that accept references as parameters, and that
// variables are properly invalidated.
class RDar9203355 {
  bool foo(unsigned valA, long long &result) const;
  bool foo(unsigned valA, int &result) const;
};
bool RDar9203355::foo(unsigned valA, int &result) const {
  long long val;
  if (foo(valA, val) ||
      (int)val != val) // no-warning
    return true;
  result = val; // no-warning
  return false;
}

// Test handling of new[].
void rdar9212512() {
  int *x = new int[10];
  for (unsigned i = 0 ; i < 2 ; ++i) {
    // This previously triggered an uninitialized values warning.
    x[i] = 1;  // no-warning
  }
}

// Test basic support for dynamic_cast<>.
struct Rdar9212495_C { virtual void bar() const; };
class Rdar9212495_B : public Rdar9212495_C {};
class Rdar9212495_A : public Rdar9212495_B {};
const Rdar9212495_A& rdar9212495(const Rdar9212495_C* ptr) {
  const Rdar9212495_A& val = dynamic_cast<const Rdar9212495_A&>(*ptr);
  
  // This is not valid C++; dynamic_cast with a reference type will throw an
  // exception if the pointer does not match the expected type. However, our
  // implementation of dynamic_cast will pass through a null pointer...or a
  // "null reference"! So this branch is actually possible.
  if (&val == 0) {
    val.bar(); // expected-warning{{Called C++ object pointer is null}}
  }
  
  return val;
}

const Rdar9212495_A* rdar9212495_ptr(const Rdar9212495_C* ptr) {
  const Rdar9212495_A* val = dynamic_cast<const Rdar9212495_A*>(ptr);

  if (val == 0) {
    val->bar(); // expected-warning{{Called C++ object pointer is null}}
  }

  return val;
}

// Test constructors invalidating arguments.  Previously this raised
// an uninitialized value warning.
extern "C" void __attribute__((noreturn)) PR9645_exit(int i);

class PR9645_SideEffect
{
public:
  PR9645_SideEffect(int *pi); // caches pi in i_
  void Read(int *pi); // copies *pi into *i_
private:
  int *i_;
};

void PR9645() {
  int i;

  PR9645_SideEffect se(&i);
  int j = 1;
  se.Read(&j); // this has a side-effect of initializing i.

  PR9645_exit(i); // no-warning
}

PR9645_SideEffect::PR9645_SideEffect(int *pi) : i_(pi) {}
void PR9645_SideEffect::Read(int *pi) { *i_ = *pi; }

// Invalidate fields during C++ method calls.
class RDar9267815 {
  int x;
  void test();
  void test_pos();
  void test2();
  void invalidate();
};

void RDar9267815::test_pos() {
  if (x == 42)
    return;
  clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}
void RDar9267815::test() {
  if (x == 42)
    return;
  if (x == 42)
    clang_analyzer_warnIfReached();  // no-warning
}

void RDar9267815::test2() {
  if (x == 42)
    return;
  invalidate();
  if (x == 42)
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
}

// Test reference parameters.
void test_ref_double_aux(double &Value);
float test_ref_double() {
  double dVal;
  test_ref_double_aux(dVal);
  // This previously warned because 'dVal' was thought to be uninitialized.
  float Val = (float)dVal; // no-warning
  return Val;
}

// Test invalidation of class fields.
class TestInvalidateClass {
public:
  int x;
};

void test_invalidate_class_aux(TestInvalidateClass &x);

int test_invalidate_class() {
  TestInvalidateClass y;
  test_invalidate_class_aux(y);
  return y.x; // no-warning
}

// Test correct pointer arithmetic using 'p--'.  This is to warn that we
// were loading beyond the written characters in buf.
char *RDar9269695(char *dst, unsigned int n)
{
  char buff[40], *p;

  p = buff;
  do
    *p++ = '0' + n % 10;
  while (n /= 10);

  do
    *dst++ = *--p; // no-warning
  while (p != buff);

  return dst;
}

// Test that we invalidate byref arguments passed to constructors.
class TestInvalidateInCtor {
public:
  TestInvalidateInCtor(unsigned &x);
};

unsigned test_invalidate_in_ctor() {
  unsigned x;
  TestInvalidateInCtor foo(x);
  return x; // no-warning
}
unsigned test_invalidate_in_ctor_new() {
  unsigned x;
  delete (new TestInvalidateInCtor(x));
  return x; // no-warning
}

// Test assigning into a symbolic offset.
struct TestAssignIntoSymbolicOffset {
  int **stuff[100];
  void test(int x, int y);
};

void TestAssignIntoSymbolicOffset::test(int x, int y) {
  x--;
  if (x > 8 || x < 0)
    return;
  if (stuff[x])
    return;
  if (!stuff[x]) {
    stuff[x] = new int*[y+1];
    // Previously triggered a null dereference.
    stuff[x][y] = 0; // no-warning
  }
}

// Test loads from static fields.  This previously triggered an uninitialized
// value warning.
class ClassWithStatic {
public:
    static const unsigned value = 1;
};

int rdar9948787_negative() {
    ClassWithStatic classWithStatic;
    unsigned value = classWithStatic.value;
    if (value == 1)
      return 1;
    clang_analyzer_warnIfReached();  // no-warning
    return 0;
}

int rdar9948787_positive() {
    ClassWithStatic classWithStatic;
    unsigned value = classWithStatic.value;
    if (value == 0)
      return 1;
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
    return 0;
}

// Regression test against global constants and switches.
enum rdar10202899_ValT { rdar10202899_ValTA, rdar10202899_ValTB, rdar10202899_ValTC };
const rdar10202899_ValT val = rdar10202899_ValTA;
void rdar10202899_test1() {
  switch (val) {
    case rdar10202899_ValTA: {}
  };
}

void rdar10202899_test2() {
  if (val == rdar10202899_ValTA)
   return;
  clang_analyzer_warnIfReached();  // no-warning
}

void rdar10202899_test3() {
  switch (val) {
    case rdar10202899_ValTA: return;
    default: ;
  };
  clang_analyzer_warnIfReached();  // no-warning
}

// This used to crash the analyzer because of the unnamed bitfield.
void PR11249()
{
  struct {
    char f1:4;
    char   :4;
    char f2[1];
    char f3;
  } V = { 1, {2}, 3 };
  if (V.f1 != 1)
    clang_analyzer_warnIfReached();  // no-warning
  if (V.f2[0] != 2)
    clang_analyzer_warnIfReached();  // no-warning
  if (V.f3 != 3)
    clang_analyzer_warnIfReached();  // no-warning
}

// Handle doing a load from the memory associated with the code for
// a function.
extern double nan( const char * );
double PR11450() {
  double NaN = *(double*) nan;
  return NaN;
}

// Test that 'this' is assumed non-null upon analyzing the entry to a "top-level"
// function (i.e., when not analyzing from a specific caller).
struct TestNullThis {
  int field;
  void test();
};

void TestNullThis::test() {
  int *p = &field;
  if (p)
    return;
  field = 2; // no-warning
}

// Test handling of 'catch' exception variables, and not warning
// about uninitialized values.
enum MyEnum { MyEnumValue };
MyEnum rdar10892489() {
  try {
      throw MyEnumValue;
  } catch (MyEnum e) {
      return e; // no-warning
  }
  return MyEnumValue;
}

MyEnum rdar10892489_positive() {
  try {
    throw MyEnumValue;
  } catch (MyEnum e) {
    int *p = 0;
    // FALSE NEGATIVE
    *p = 0xDEADBEEF; // {{null}}
    return e;
  }
  return MyEnumValue;
}

// Test handling of catch with no condition variable.
void PR11545() {
  try
  {
      throw;
  }
  catch (...)
  {
  }
}

void PR11545_positive() {
  try
  {
      throw;
  }
  catch (...)
  {
    int *p = 0;
    // FALSE NEGATIVE
    *p = 0xDEADBEEF; // {{null}}
  }
}

// Test handling taking the address of a field.  While the analyzer
// currently doesn't do anything intelligent here, this previously
// resulted in a crash.
class PR11146 {
public:
  struct Entry;
  void baz();
};

struct PR11146::Entry {
  int x;
};

void PR11146::baz() {
  (void) &Entry::x;
}

// Test symbolicating a reference.  In this example, the
// analyzer (originally) didn't know how to handle x[index - index2],
// returning an UnknownVal.  The conjured symbol wasn't a location,
// and would result in a crash.
void rdar10924675(unsigned short x[], int index, int index2) {
  unsigned short &y = x[index - index2];
  if (y == 0)
    return;
}

// Test handling CXXScalarValueInitExprs.
void rdar11401827() {
  int x = int();
  if (!x) {
    clang_analyzer_warnIfReached();  // expected-warning{{REACHABLE}}
    ; // Suppress warning that both branches are identical
  }
  else {
    clang_analyzer_warnIfReached();  // no-warning
  }
}

//===---------------------------------------------------------------------===//
// Handle inlining of C++ method calls.
//===---------------------------------------------------------------------===//

struct A {
  int *p;
  void foo(int *q) {
    p = q;
  }
  void bar() {
    *p = 0; // expected-warning {{null pointer}}
  }
};

void test_inline() {
  A a;
  a.foo(0);
  a.bar();
}

void test_alloca_in_a_recursive_function(int p1) {
    __builtin_alloca (p1);
    test_alloca_in_a_recursive_function(1);
    test_alloca_in_a_recursive_function(2);
}

//===---------------------------------------------------------------------===//
// Random tests.
//===---------------------------------------------------------------------===//

// Tests assigning using a C-style initializer to a struct
// variable whose sub-field is also a struct.  This currently
// results in a CXXTempObjectRegion being created, but not
// properly handled.  For now, we just ignore that value
// to avoid a crash (<rdar://problem/12753384>).
struct RDar12753384_ClassA {
  unsigned z;
};
struct  RDar12753384_ClassB {
  unsigned x;
  RDar12753384_ClassA y[ 8 ] ;
};
unsigned RDar12753384() {
  RDar12753384_ClassB w = { 0x00 };
  RDar12753384_ClassA y[8];
  return w.x;
}

// This testcase tests whether we treat the anonymous union and union
// the same way.  This previously resulted in a "return of stack address"
// warning because the anonymous union resulting in a temporary object
// getting put into the initializer.  We still aren't handling this correctly,
// but now if a temporary object appears in an initializer we just ignore it.
// Fixes <rdar://problem/12755044>.

struct Rdar12755044_foo
{
    struct Rdar12755044_bar
    {
        union baz
        {
            int   i;
        };
    } aBar;
};

struct Rdar12755044_foo_anon
{
    struct Rdar12755044_bar
    {
        union
        {
            int   i;
        };
    } aBar;
};

const Rdar12755044_foo_anon *radar12755044_anon() {
  static const Rdar12755044_foo_anon Rdar12755044_foo_list[] = { { { } } };
  return Rdar12755044_foo_list; // no-warning
}

const Rdar12755044_foo *radar12755044() {
  static const Rdar12755044_foo Rdar12755044_foo_list[] = { { { } } };
  return Rdar12755044_foo_list; // no-warning
}

// Test the correct handling of integer to bool conversions.  Previously
// this resulted in a false positive because integers were being truncated
// and not tested for non-zero.
void rdar12759044() {
  int flag = 512;
  if (!(flag & 512)) {
    clang_analyzer_warnIfReached();  // no-warning
  }
}

// The analyzer currently does not model complex types.  Test that the load
// from 'x' is not flagged as being uninitialized.
typedef __complex__ float _ComplexT;
void rdar12964481(_ComplexT *y) {
   _ComplexT x;
   __real__ x = 1.0;
   __imag__ x = 1.0;
   *y *= x; // no-warning
}
void rdar12964481_b(_ComplexT *y) {
   _ComplexT x;
   // Eventually this should be a warning.
   *y *= x; // no-warning
}

// Test case for PR 12921.  This previously produced
// a bogus warning.
static const int pr12921_arr[] = { 0, 1 };
static const int pr12921_arrcount = sizeof(pr12921_arr)/sizeof(int);

int pr12921(int argc, char **argv) {
  int i, retval;
  for (i = 0; i < pr12921_arrcount; i++) {
    if (argc == i) {
      retval = i;
      break;
    }
  }

  // No match
  if (i == pr12921_arrcount) return 66;
  return pr12921_arr[retval];
}

