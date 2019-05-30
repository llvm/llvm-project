// RUN: %clang_analyze_cc1 -analyzer-store=region -analyzer-checker=core,unix.Malloc,debug.ExprInspection -analyzer-output=text -verify %s

typedef unsigned int NSUInteger;
typedef __typeof__(sizeof(int)) size_t;

void *malloc(size_t);
void *calloc(size_t nmemb, size_t size);
void free(void *);

void clang_analyzer_eval(int);

struct s {
  int data;
};

struct s global;

void g(int);

void f4() {
  int a;
  if (global.data == 0)
    a = 3;
  if (global.data == 0) // When the true branch is feasible 'a = 3'.
    g(a); // no-warning
}


// Test uninitialized value due to part of the structure being uninitialized.
struct TestUninit { int x; int y; };
struct TestUninit test_uninit_aux();
void test_unit_aux2(int);
void test_uninit_pos() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2 = test_uninit_aux();
  int z; // expected-note{{'z' declared without an initial value}}
  v1.y = z; // expected-warning{{Assigned value is garbage or undefined}}
            // expected-note@-1{{Assigned value is garbage or undefined}}
  test_unit_aux2(v2.x + v1.y);
}
void test_uninit_pos_2() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2;
  test_unit_aux2(v2.x + v1.y);  // expected-warning{{The left operand of '+' is a garbage value}}
                                // expected-note@-1{{The left operand of '+' is a garbage value}}
}
void test_uninit_pos_3() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2;
  test_unit_aux2(v1.y + v2.x);  // expected-warning{{The right operand of '+' is a garbage value}}
                                // expected-note@-1{{The right operand of '+' is a garbage value}}
}

void test_uninit_neg() {
  struct TestUninit v1 = { 0, 0 };
  struct TestUninit v2 = test_uninit_aux();
  test_unit_aux2(v2.x + v1.y);
}

extern void test_uninit_struct_arg_aux(struct TestUninit arg);
void test_uninit_struct_arg() {
  struct TestUninit x; // expected-note{{'x' initialized here}}
  test_uninit_struct_arg_aux(x); // expected-warning{{Passed-by-value struct argument contains uninitialized data (e.g., field: 'x')}}
                                 // expected-note@-1{{Passed-by-value struct argument contains uninitialized data (e.g., field: 'x')}}
}

@interface Foo
- (void) passVal:(struct TestUninit)arg;
@end
void testFoo(Foo *o) {
  struct TestUninit x; // expected-note{{'x' initialized here}}
  [o passVal:x]; // expected-warning{{Passed-by-value struct argument contains uninitialized data (e.g., field: 'x')}}
                 // expected-note@-1{{Passed-by-value struct argument contains uninitialized data (e.g., field: 'x')}}
}

// Test case from <rdar://problem/7780304>.  That shows an uninitialized value
// being used in the LHS of a compound assignment.
void rdar_7780304() {
  typedef struct s_r7780304 { int x; } s_r7780304;
  s_r7780304 b;
  b.x |= 1; // expected-warning{{The left expression of the compound assignment is an uninitialized value. The computed value will also be garbage}}
            // expected-note@-1{{The left expression of the compound assignment is an uninitialized value. The computed value will also be garbage}}
}


// The flip side of PR10163 -- float arrays that are actually uninitialized
void test_PR10163(float);
void PR10163 (void) {
  float x[2];
  test_PR10163(x[1]); // expected-warning{{uninitialized value}}
                      // expected-note@-1{{1st function call argument is an uninitialized value}}
}

// PR10163 -- don't warn for default-initialized float arrays.
void PR10163_default_initialized_arrays(void) {
  float x[2] = {0};
  test_PR10163(x[1]); // no-warning  
}

struct MyStr {
  int x;
  int y;
};
void swap(struct MyStr *To, struct MyStr *From) {
  // This is not really a swap but close enough for our test.
  To->x = From->x;
  To->y = From->y; // expected-note{{Uninitialized value stored to field 'y'}}
}
int test_undefined_member_assignment_in_swap(struct MyStr *s2) {
  struct MyStr s1;
  s1.x = 5;
  swap(s2, &s1); // expected-note{{Calling 'swap'}}
                 // expected-note@-1{{Returning from 'swap'}}
  return s2->y; // expected-warning{{Undefined or garbage value returned to caller}}
                // expected-note@-1{{Undefined or garbage value returned to caller}}
}

@interface A
- (NSUInteger)foo;
@end

NSUInteger f8(A* x){
  const NSUInteger n = [x foo];
  int* bogus;  

  if (n > 0) {    // tests const cast transfer function logic
    NSUInteger i;
    
    for (i = 0; i < n; ++i)
      bogus = 0;

    if (bogus)  // no-warning
      return n+1;
  }
  
  return n;
}




typedef struct {
  float x;
  float y;
  float z;
} Point;
typedef struct {
  Point origin;
  int size;
} Circle;

Point makePoint(float x, float y) {
  Point result;
  result.x = x;
  result.y = y;
  result.z = 0.0;
  return result;
}

void PR14765_test() {
  Circle *testObj = calloc(sizeof(Circle), 1);

  clang_analyzer_eval(testObj->size == 0); // expected-warning{{TRUE}}
                                           // expected-note@-1{{TRUE}}

  testObj->origin = makePoint(0.0, 0.0);
  if (testObj->size > 0) { ; } // expected-note{{Assuming field 'size' is <= 0}}
                               // expected-note@-1{{Taking false branch}}

  // FIXME: Assigning to 'testObj->origin' kills the default binding for the
  // whole region, meaning that we've forgotten that testObj->size should also
  // default to 0. Tracked by <rdar://problem/12701038>.
  // This should be TRUE.
  clang_analyzer_eval(testObj->size == 0); // expected-warning{{UNKNOWN}}
                                           // expected-note@-1{{UNKNOWN}}

  free(testObj);
}

void PR14765_argument(Circle *testObj) {
  int oldSize = testObj->size;
  clang_analyzer_eval(testObj->size == oldSize); // expected-warning{{TRUE}}
                                                 // expected-note@-1{{TRUE}}

  testObj->origin = makePoint(0.0, 0.0);
  clang_analyzer_eval(testObj->size == oldSize); // expected-warning{{TRUE}}
                                                 // expected-note@-1{{TRUE}}
}


typedef struct {
  int x;
  int y;
  int z;
} IntPoint;
typedef struct {
  IntPoint origin;
  int size;
} IntCircle;

IntPoint makeIntPoint(int x, int y) {
  IntPoint result;
  result.x = x;
  result.y = y;
  result.z = 0;
  return result;
}

void PR14765_test_int() {
  IntCircle *testObj = calloc(sizeof(IntCircle), 1);

  clang_analyzer_eval(testObj->size == 0); // expected-warning{{TRUE}}
                                           // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.x == 0); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.y == 0); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.z == 0); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}

  testObj->origin = makeIntPoint(1, 2);
  if (testObj->size > 0) { ; } // expected-note{{Assuming field 'size' is <= 0}}
                               // expected-note@-1{{Taking false branch}}
                               // expected-note@-2{{Assuming field 'size' is <= 0}}
                               // expected-note@-3{{Taking false branch}}
                               // expected-note@-4{{Assuming field 'size' is <= 0}}
                               // expected-note@-5{{Taking false branch}}
                               // expected-note@-6{{Assuming field 'size' is <= 0}}
                               // expected-note@-7{{Taking false branch}}

  // FIXME: Assigning to 'testObj->origin' kills the default binding for the
  // whole region, meaning that we've forgotten that testObj->size should also
  // default to 0. Tracked by <rdar://problem/12701038>.
  // This should be TRUE.
  clang_analyzer_eval(testObj->size == 0); // expected-warning{{UNKNOWN}}
                                           // expected-note@-1{{UNKNOWN}}
  clang_analyzer_eval(testObj->origin.x == 1); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.y == 2); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.z == 0); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}

  free(testObj);
}

void PR14765_argument_int(IntCircle *testObj) {
  int oldSize = testObj->size;
  clang_analyzer_eval(testObj->size == oldSize); // expected-warning{{TRUE}}
                                                 // expected-note@-1{{TRUE}}

  testObj->origin = makeIntPoint(1, 2);
  clang_analyzer_eval(testObj->size == oldSize); // expected-warning{{TRUE}}
                                                 // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.x == 1); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.y == 2); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.z == 0); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
}


void rdar13292559(Circle input) {
  extern void useCircle(Circle);

  Circle obj = input;
  useCircle(obj); // no-warning

  // This generated an "uninitialized 'size' field" warning for a (short) while.
  obj.origin = makePoint(0.0, 0.0);
  useCircle(obj); // no-warning
}


typedef struct {
  int x;
  int y;
} IntPoint2D;
typedef struct {
  IntPoint2D origin;
  int size;
} IntCircle2D;

IntPoint2D makeIntPoint2D(int x, int y) {
  IntPoint2D result;
  result.x = x;
  result.y = y;
  return result;
}

void testSmallStructsCopiedPerField() {
  IntPoint2D a;
  a.x = 0;

  IntPoint2D b = a;
  extern void useInt(int);
  useInt(b.x); // no-warning
  useInt(b.y); // expected-warning{{uninitialized}}
               // expected-note@-1{{uninitialized}}
}

void testLargeStructsNotCopiedPerField() {
  IntPoint a;
  a.x = 0;

  IntPoint b = a;
  extern void useInt(int);
  useInt(b.x); // no-warning
  useInt(b.y); // no-warning
}

void testSmallStructInLargerStruct() {
  IntCircle2D *testObj = calloc(sizeof(IntCircle2D), 1);

  clang_analyzer_eval(testObj->size == 0); // expected-warning{{TRUE}}
                                           // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.x == 0); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.y == 0); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}

  testObj->origin = makeIntPoint2D(1, 2);
  if (testObj->size > 0) { ; } // expected-note{{Field 'size' is <= 0}}
                               // expected-note@-1{{Taking false branch}}
                               // expected-note@-2{{Field 'size' is <= 0}}
                               // expected-note@-3{{Taking false branch}}
                               // expected-note@-4{{Field 'size' is <= 0}}
                               // expected-note@-5{{Taking false branch}}

  clang_analyzer_eval(testObj->size == 0); // expected-warning{{TRUE}}
                                           // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.x == 1); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.y == 2); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}

  free(testObj);
}

void testCopySmallStructIntoArgument(IntCircle2D *testObj) {
  int oldSize = testObj->size;
  clang_analyzer_eval(testObj->size == oldSize); // expected-warning{{TRUE}}
                                                 // expected-note@-1{{TRUE}}

  testObj->origin = makeIntPoint2D(1, 2);
  clang_analyzer_eval(testObj->size == oldSize); // expected-warning{{TRUE}}
                                                 // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.x == 1); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
  clang_analyzer_eval(testObj->origin.y == 2); // expected-warning{{TRUE}}
                                               // expected-note@-1{{TRUE}}
}

void testSmallStructBitfields() {
  struct {
    int x : 4;
    int y : 4;
  } a, b;

  a.x = 1;
  a.y = 2;

  b = a;
  clang_analyzer_eval(b.x == 1); // expected-warning{{TRUE}}
                                 // expected-note@-1{{TRUE}}
  clang_analyzer_eval(b.y == 2); // expected-warning{{TRUE}}
                                 // expected-note@-1{{TRUE}}
}

void testSmallStructBitfieldsFirstUndef() {
  struct {
    int x : 4;
    int y : 4;
  } a, b;

  a.y = 2;

  b = a;
  clang_analyzer_eval(b.y == 2); // expected-warning{{TRUE}}
                                 // expected-note@-1{{TRUE}}
  clang_analyzer_eval(b.x == 1); // expected-warning{{garbage}}
                                 // expected-note@-1{{garbage}}
}

void testSmallStructBitfieldsSecondUndef() {
  struct {
    int x : 4;
    int y : 4;
  } a, b;

  a.x = 1;

  b = a;
  clang_analyzer_eval(b.x == 1); // expected-warning{{TRUE}}
                                 // expected-note@-1{{TRUE}}
  clang_analyzer_eval(b.y == 2); // expected-warning{{garbage}}
                                 // expected-note@-1{{garbage}}
}

void testSmallStructBitfieldsFirstUnnamed() {
  struct {
    int : 4;
    int y : 4;
  } a, b, c; // expected-note{{'c' initialized here}}

  a.y = 2;

  b = a;
  clang_analyzer_eval(b.y == 2); // expected-warning{{TRUE}}
                                 // expected-note@-1{{TRUE}}

  b = c; // expected-note{{Uninitialized value stored to 'b.y'}}
  clang_analyzer_eval(b.y == 2); // expected-warning{{garbage}}
                                 // expected-note@-1{{garbage}}
}

void testSmallStructBitfieldsSecondUnnamed() {
  struct {
    int x : 4;
    int : 4;
  } a, b, c; // expected-note{{'c' initialized here}}

  a.x = 1;

  b = a;
  clang_analyzer_eval(b.x == 1); // expected-warning{{TRUE}}
                                 // expected-note@-1{{TRUE}}

  b = c; // expected-note{{Uninitialized value stored to 'b.x'}}
  clang_analyzer_eval(b.x == 1); // expected-warning{{garbage}}
                                 // expected-note@-1{{garbage}}
}
