// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,deadcode.DeadStores,osx.cocoa.RetainCount,unix.Malloc,unix.MismatchedDeallocator -analyzer-output=plist -o %t -w %s
// RUN: %normalize_plist <%t | diff -ub %S/Inputs/expected-plists/edges-new.mm.plist -

//===----------------------------------------------------------------------===//
// Forward declarations (from headers).
//===----------------------------------------------------------------------===//

typedef const struct __CFNumber * CFNumberRef;
typedef const struct __CFAllocator * CFAllocatorRef;
extern const CFAllocatorRef kCFAllocatorDefault;
typedef signed long CFIndex;
enum {
  kCFNumberSInt8Type = 1,
  kCFNumberSInt16Type = 2,
  kCFNumberSInt32Type = 3,
  kCFNumberSInt64Type = 4,
  kCFNumberFloat32Type = 5,
  kCFNumberFloat64Type = 6,
  kCFNumberCharType = 7,
  kCFNumberShortType = 8,
  kCFNumberIntType = 9,
  kCFNumberLongType = 10,
  kCFNumberLongLongType = 11,
  kCFNumberFloatType = 12,
  kCFNumberDoubleType = 13,
  kCFNumberCFIndexType = 14,
  kCFNumberNSIntegerType = 15,
  kCFNumberCGFloatType = 16,
  kCFNumberMaxType = 16
};
typedef CFIndex CFNumberType;
CFNumberRef CFNumberCreate(CFAllocatorRef allocator, CFNumberType theType, const void *valuePtr);

#define nil ((id)0)

__attribute__((objc_root_class))
@interface NSObject
+ (instancetype) alloc;
- (instancetype) init;
- (instancetype)retain;
- (void)release;
@end

@interface NSArray : NSObject
@end

//===----------------------------------------------------------------------===//
// Basic tracking of null and tests for null.
//===----------------------------------------------------------------------===//

void test_null_init(void) {
  int *p = 0;
  *p = 0xDEADBEEF;
}

void test_null_assign(void) {
  int *p;
  p = 0;
  *p = 0xDEADBEEF;
}

void test_null_assign_transitive(void) {
  int *p;
  p = 0;
  int *q = p;
  *q = 0xDEADBEEF;
}

void test_null_cond(int *p) {
  if (!p) {
    *p = 0xDEADBEEF;
  }
}

void test_null_cond_transitive(int *q) {
  if (!q) {
    int *p = q;
    *p = 0xDEADBEEF;
  }
}

void test_null_field(void) {
  struct s { int *p; } x;
  x.p = 0;
  *(x.p) = 0xDEADBEEF;
}

void test_assumptions(int a, int b)
{
  if (a == 0) {
    return;
  }
  if (b != 0) {
    return;
  }
  int *p = 0;
  *p = 0xDEADBEEF;
}

int *bar_cond_assign();
int test_cond_assign() {
  int *p;
  if ((p = bar_cond_assign()))
    return 1;
  return *p;
}

//===----------------------------------------------------------------------===//
// Diagnostics for leaks and "noreturn" paths.
//===----------------------------------------------------------------------===//


// <rdar://problem/8331641> leak reports should not show paths that end with exit() (but ones that don't end with exit())

void stop() __attribute__((noreturn));

void rdar8331641(int x) {
  signed z = 1;
  CFNumberRef value = CFNumberCreate(kCFAllocatorDefault, kCFNumberSInt32Type, &z); // expected-warning{{leak}}
  if (x)
    stop();
  (void) value;
}

//===----------------------------------------------------------------------===//
// Test loops and control-flow.
//===----------------------------------------------------------------------===//

void test_objc_fast_enumeration(NSArray *x) {
  id obj;
  for (obj in x)
    *(volatile int *)0 = 0;
}

void test_objc_fast_enumeration_2(id arr) {
  int x;
  for (id obj in arr) {
    x = 1;
  }
  x += 1;
}

// Test that loops are documented in the path.
void rdar12280665() {
  for (unsigned i = 0; i < 2; ++i) {
	  if (i == 1) {
		  int *p = 0;
		  *p = 0xDEADBEEF; // expected-warning {{dereference}}
	  }
  }
}

// Test for a "loop executed 0 times" diagnostic.
int *radar12322528_bar();

void radar12322528_for(int x) {
  int z;
  int *p = 0;
  for (unsigned i = 0; i < x; ++i) {
    p = radar12322528_bar();
  }
  *p = 0xDEADBEEF;
}

void radar12322528_while(int x) {
  int *p = 0;
  unsigned i = 0;
  for ( ; i < x ; ) {
    ++i;
    p = radar12322528_bar();
  }
  *p = 0xDEADBEEF;
}

void radar12322528_foo_2() {
  int *p = 0;
  for (unsigned i = 0; i < 2; ++i) {
    if (i == 0)
      continue;

    if (i == 1) {

      break;
    }
  }
  *p = 0xDEADBEEF;
}

void test_loop_diagnostics() {
  int *p = 0;
  for (int i = 0; i < 2; ++i) { p = 0; }
  *p = 1;
}

void test_loop_diagnostics_2() {
  int *p = 0;

  for (int i = 0; i < 2; ) {

    ++i;

    p = 0;

  }

  *p = 1;
}

void test_loop_diagnostics_3() {
  int z;
  int y;
  int k;
  int *p = 0;
  int i = 0;
  while (i < 2) {
    ++i;
    p = 0;
  }
  * p = 1;
}

void test_do_while() {
  unsigned i = 0;

  int *p;

  do {

    ++i;
    p = 0;

  } while (i< 2);

  *p = 0xDEADBEEF;
}


void test_logical_and() {
  int *p = 0;
  if (1 && 2) {
    *p = 0xDEADBEEF;
  }
}

void test_logical_or() {
  int *p = 0;
  if (0 || 2) {
    *p = 0xDEADBEEF;
  }
}

void test_logical_or_call() {
  extern int call(int);
  int *p = 0;
  if (call(0 || 2)) {
    *p = 0xDEADBEEF;
  }
}

void test_nested_logicals(int coin) {
  int *p = 0;

  if ((0 || 0) || coin) {
    *p = 0xDEADBEEF;
  }

  if (0 || (0 || !coin)) {
    *p = 0xDEADBEEF;
  }
}

void test_deeply_nested_logicals() {
  extern int call(int);
  int *p = 0;

  if ((0 || (5 && 0)) ? 0 : ((0 || 4) ? call(1 && 5) : 0)) {

    *p = 0xDEADBEEF;
  }
}

void test_ternary(int x, int *y) {
  int z = x ? 0 : 1;

  int *p = z ? y : 0;

  *p = 0xDEADBEEF;
}

void testUseless(int *y) {
  if (y) {

  }
  if (y) {

  }
  int *p = 0;
  *p = 0xDEADBEEF;
}

//===----------------------------------------------------------------------===//
// Interprocedural tests.
//===----------------------------------------------------------------------===//

@interface IPA_Foo
- (int *) returnsPointer;
@end

int testFoo(IPA_Foo *x) {
  if (x)
    return 1;
  return *[x returnsPointer];
}

@interface IPA_X : NSObject
- (int *)getPointer;
@end

void test1_IPA_X() {
  IPA_X *x = nil;
  *[x getPointer] = 1; // here
}


@interface IPA_Y : NSObject
- (IPA_Y *)opaque;
- (IPA_X *)getX;
@end

@implementation IPA_Y
- (IPA_X *)getX {
  return nil;
}
@end

void test_IPA_Y(IPA_Y *y) {
  if (y)
    return;

  IPA_X *x = [[y opaque] getX]; // here
  *[x getPointer] = 1;
}

// From diagnostics/report-issues-within-main-file.cpp:
void causeDivByZeroInMain(int in) {
  int m = 0;
  m = in/m;
  m++;
}

void mainPlusMain() {
  int i = 0;
  i++;
  causeDivByZeroInMain(i);
  i++;
}

// From inlining/path-notes.c:
int *getZero() {
  int *p = 0;
  return p;
}

void usePointer(int *p) {
  *p = 1;
}

void testUseOfNullPointer() {
  // Test the case where an argument expression is itself a call.
  usePointer(getZero());
}


//===----------------------------------------------------------------------===//
// Misc. tests.
//===----------------------------------------------------------------------===//

// Test for tracking null state of ivars.
@interface RDar12114812 : NSObject  { char *p; }
@end
@implementation RDar12114812
- (void)test {
  p = 0;
  *p = 1;
}
@end

// Test diagnostics for initialization of structs.
void RDar13295437_f(void *i) __attribute__((__nonnull__));
struct RDar13295437_S { int *i; };
int  RDar13295437() {
  struct RDar13295437_S s = {0};
  struct RDar13295437_S *sp = &s;
  RDar13295437_f(sp->i);
  return 0;
}


void testCast(int coin) {
  if (coin) {
    (void)(1+2);
    (void)(2+3);
    (void)(3+4);
    *(volatile int *)0 = 1;
  }
}

// The following previously crashed when generating extensive diagnostics.
// <rdar://problem/10797980>
@interface RDar10797980_help
@property (readonly) int x;
@end
@interface RDar10797980 : NSObject {
  RDar10797980_help *y;
}
- (void) test;
@end
@implementation RDar10797980
- (void) test {
  if (y.x == 1) {
    int *p = 0;
    *p = 0xDEADBEEF; // expected-warning {{deference}}
  }
}

// The original source for the above Radar contains another problem:
// if the end-of-path node is an implicit statement, it may not have a valid
// source location. <rdar://problem/12446776>
- (void)test2 {
  if (bar_cond_assign()) {
    id foo = [[RDar10797980 alloc] init]; // leak
  }
  (void)y; // first statement after the 'if' is an implicit 'self' DeclRefExpr
}

@end

void variousLoops(id input) {
  extern int a();
  extern int b();
  extern int c();

  extern int work();

  while (a()) {
    work();
    work();
    work();
    *(volatile int *)0 = 1;
  }

  int first = 1;
  do {
    work();
    work();
    work();
    if (!first)
      *(volatile int *)0 = 1;
    first = 0;
  } while (a());

  for (int i = 0; i != b(); ++i) {
    work();
    *(volatile int *)0 = 1;
  }

  for (id x in input) {
    work();
    work();
    work();
    (void)x;
    *(volatile int *)0 = 1;
  }

  int z[] = {1,2};
  for (int y : z) {
    work();
    work();
    work();
    (void)y;
  }

  int empty[] = {};
  for (int y : empty) {
    work();
    work();
    work();
    (void)y;
  }

  for (int i = 0; ; ++i) {
    work();
    if (i == b())
      break;
  }

  int i;
  for (i = 0; i != b(); ++i) {
    work();
    *(volatile int *)0 = 1;
  }

  for (; i != b(); ++i) {
    work();
    *(volatile int *)0 = 1;
  }

  for (; i != b(); ) {
    work();
    if (i == b())
      break;
    *(volatile int *)0 = 1;
  }

  for (;;) {
    work();
    if (i == b())
      break;
  }

  *(volatile int *)0 = 1;
}

void *malloc(unsigned long);
void *realloc(void *, unsigned long);
void free(void *);

void reallocDiagnostics() {
  char * buf = (char*)malloc(100);
  char * tmp;
  tmp = (char*)realloc(buf, 0x1000000);
  if (!tmp) {
    return;// expected-warning {{leak}}
  }
  buf = tmp;
  free(buf);
}

template <typename T>
class unique_ptr {
  T *ptr;
public:
  explicit unique_ptr(T *p) : ptr(p) {}
  ~unique_ptr() { delete ptr; }
};

void test() {
  int i = 0;
  ++i;

  unique_ptr<int> p(new int[4]);
  {
    ++i;
  }
}

void longLines() {
  id foo = [[NSObject alloc] init]; // leak
  id bar =
           [foo retain];
  [bar release];
  id baz = [foo
              retain];
  [baz release];
  // This next line is intentionally longer than 80 characters.
  id garply = [foo                                                              retain];
  [garply release];
}

#define POINTER(T) T*
POINTER(void) testMacroInFunctionDecl(void *q) {
  int *p = 0;
  *p = 1;
  return q;
}

namespace rdar14960554 {
  class Foo {
    int a = 1;
    int b = 2;
    int c = 3;

    Foo() :
      a(0),
      c(3) {
      // Check that we don't have an edge to the in-class initializer for 'b'.
      if (b == 2)
        *(volatile int *)0 = 1;
    }
  };
}

