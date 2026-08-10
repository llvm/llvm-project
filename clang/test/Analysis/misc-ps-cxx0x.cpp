// RUN: %clang_analyze_cc1 -analyzer-checker=core.NullDereference,core.uninitialized.UndefReturn -std=c++11 %s -verify -o /dev/null

void test_static_assert() {
  static_assert(sizeof(void *) == sizeof(void*), "test_static_assert");
}

void test_analyzer_working() {
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}

// Test that pointer-to-member functions don't cause the analyzer
// to crash.
struct RDar10243398 {
  void bar(int x);
};

typedef void (RDar10243398::*RDar10243398MemberFn)(int x);

void test_rdar10243398(RDar10243398 *p) {
  RDar10243398MemberFn q = &RDar10243398::bar;
  ((*p).*(q))(1);
}

// Tests for CXXTemporaryObjectExpr.
struct X {
    X( int *ip, int );
};

// Test to see if CXXTemporaryObjectExpr is being handled.
int tempobj1()
{
  int j;
  int i;
  X a = X( &j, 1 );

  return i; // expected-warning {{Undefined or garbage value returned to caller}}
}

// Test to see if CXXTemporaryObjectExpr invalidates arguments.
int tempobj2()
{
  int j;
  X a = X( &j, 1 );

  return j; // no-warning
}


// Test for correct handling of C++ ForRange statement.
void test1() {
  int array[2] = { 1, 2 };
  int j = 0;
  for ( int i : array )
    j += i;
  int *p = 0;
  *p = 0xDEADBEEF;  // expected-warning {{null}}
}

void test2() {
  int array[2] = { 1, 2 };
  int j = 0;
  for (int i : array)
    j += i;
  if (j == 3)
    return;
  int *p = 0;
  *p = 0xDEADBEEF;  // no-warning
}

// Do not crash on the following when constructing the
// callgraph.
struct RDar11178609 {
  ~RDar11178609() = delete;
};

// Tests that dynamic_cast handles references to C++ classes.  Previously
// this crashed.
class rdar11817693_BaseBase {};
class rdar11817693_BaseInterface {};
class rdar11817693_Base : public rdar11817693_BaseBase, public rdar11817693_BaseInterface {};
class rdar11817693 : public rdar11817693_Base {
  virtual void operator=(const rdar11817693_BaseBase& src);
  void operator=(const rdar11817693& src);
};
void rdar11817693::operator=(const rdar11817693& src) {
  operator=(dynamic_cast<const rdar11817693_BaseBase&>(src));
}

