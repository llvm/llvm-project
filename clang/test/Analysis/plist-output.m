// RUN: %clang_analyze_cc1 -analyzer-config eagerly-assume=false %s -analyzer-checker=osx.cocoa.RetainCount,deadcode.DeadStores,core -analyzer-output=plist -o %t.plist
// RUN: cat %t.plist | %diff_plist %S/Inputs/expected-plists/plist-output.m.plist

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
  if (p = bar_cond_assign())
    return 1;
  return *p;
}

// The following previously crashed when generating extensive diagnostics.
// <rdar://problem/10797980>
@interface RDar10797980_help
@property (readonly) int x;
@end

@interface RDar10797980 {
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
    if (i == 1)
      break;
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
  int *p = 0;
  int i = 0;
  while (i < 2) {
    ++i;
    p = 0;
  }
  *p = 1;
}

void test_loop_fast_enumeration(id arr) {
  int x;
  for (id obj in arr) {
    x = 1;
  }
  x += 1;
}

@interface RDar12114812 { char *p; }
@end

@implementation RDar12114812
- (void)test {
  p = 0;
  *p = 1;
}
@end

// Test diagnostics for initialization of structs.
void RDar13295437_f(void *i) __attribute__((__nonnull__));

struct  RDar13295437_S { int *i; };

int  RDar13295437() {
  struct RDar13295437_S s = {0};
  struct RDar13295437_S *sp = &s;
  RDar13295437_f(sp->i);
}

@interface Foo
- (int *) returnsPointer;
@end

int testFoo(Foo *x) {
  if (x)
    return 1;
  return *[x returnsPointer];
}

