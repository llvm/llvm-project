// RUN: %clang_analyze_cc1 -triple i386-apple-darwin9 -analyzer-checker=core,alpha.core.CastToStruct,alpha.security.ReturnPtrRange,alpha.security.ArrayBound -analyzer-store=region -verify -fblocks -analyzer-opt-analyze-nested-blocks -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin9 -DTEST_64 -analyzer-checker=core,alpha.core.CastToStruct,alpha.security.ReturnPtrRange,alpha.security.ArrayBound -analyzer-store=region -verify -fblocks   -analyzer-opt-analyze-nested-blocks -Wno-objc-root-class %s

typedef long unsigned int size_t;
void *memcpy(void *, const void *, size_t);
void *alloca(size_t);

typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject  - (BOOL)isEqual:(id)object; @end
@protocol NSCopying  - (id)copyWithZone:(NSZone *)zone; @end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding  - (void)encodeWithCoder:(NSCoder *)aCoder; @end
@interface NSObject <NSObject> {} - (id)init; @end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end extern NSString * const NSBundleDidLoadNotification;
@interface NSAssertionHandler : NSObject {}
+ (NSAssertionHandler *)currentHandler;
- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
@end
extern NSString * const NSConnectionReplyMode;

#ifdef TEST_64
typedef long long int64_t;
typedef int64_t intptr_t;
#else
typedef int int32_t;
typedef int32_t intptr_t;
#endif

//---------------------------------------------------------------------------
// Test case 'checkaccess_union' differs for region store and basic store.
// The basic store doesn't reason about compound literals, so the code
// below won't fire an "uninitialized value" warning.
//---------------------------------------------------------------------------

// PR 2948 (testcase; crash on VisitLValue for union types)
// http://llvm.org/bugs/show_bug.cgi?id=2948
void checkaccess_union() {
  int ret = 0, status;
  // Since RegionStore doesn't handle unions yet,
  // this branch condition won't be triggered
  // as involving an uninitialized value.  
  if (((((__extension__ (((union {  // no-warning
    __typeof (status) __in; int __i;}
    )
    {
      .__in = (status)}
      ).__i))) & 0xff00) >> 8) == 1)
        ret = 1;
}

// Check our handling of fields being invalidated by function calls.
struct test2_struct { int x; int y; char* s; };
void test2_help(struct test2_struct* p);

char test2() {
  struct test2_struct s;
  test2_help(&s);
  char *p = 0;
  
  if (s.x > 1) {
    if (s.s != 0) {
      p = "hello";
    }
  }
  
  if (s.x > 1) {
    if (s.s != 0) {
      return *p;
    }
  }

  return 'a';
}

// BasicStore handles this case incorrectly because it doesn't reason about
// the value pointed to by 'x' and thus creates different symbolic values
// at the declarations of 'a' and 'b' respectively.  RegionStore handles
// it correctly. See the companion test in 'misc-ps-basic-store.m'.
void test_trivial_symbolic_comparison_pointer_parameter(int *x) {
  int a = *x;
  int b = *x;
  if (a != b) {
    int *p = 0;
    *p = 0xDEADBEEF;     // no-warning
  }
}

// This is a modified test from 'misc-ps.m'.  Here we have the extra
// NULL dereferences which are pruned out by RegionStore's symbolic reasoning
// of fields.
typedef struct _BStruct { void *grue; } BStruct;
void testB_aux(void *ptr);

void testB(BStruct *b) {
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    int __gruev2__ = *__gruep__;
    if (__gruev__ != __gruev2__) {
      int *p = 0;
      *p = 0xDEADBEEF; // no-warning
    }

    testB_aux(__gruep__);
  }
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    int __gruev2__ = *__gruep__;
    if (__gruev__ != __gruev2__) {
      int *p = 0;
      *p = 0xDEADBEEF; // no-warning
    }

    if (~0 != __gruev__) {}
  }
}

void testB_2(BStruct *b) {
  {
    int **__gruep__ = ((int **)&((b)->grue));
    int *__gruev__ = *__gruep__;
    testB_aux(__gruep__);
  }
  {
    int **__gruep__ = ((int **)&((b)->grue));
    int *__gruev__ = *__gruep__;
    if ((int*)~0 != __gruev__) {}
  }
}

// This test case is a reduced case of a caching bug discovered by an
// assertion failure in RegionStoreManager::BindArray.  Essentially the
// DeclStmt is evaluated twice, but on the second loop iteration the
// engine caches out.  Previously a false transition would cause UnknownVal
// to bind to the variable, firing an assertion failure.  This bug was fixed
// in r76262.
void test_declstmt_caching() {
again:
  {
    const char a[] = "I like to crash";
    goto again;
  }
}

//===----------------------------------------------------------------------===//
// Reduced test case from <rdar://problem/7114618>.
// Basically a null check is performed on the field value, which is then
// assigned to a variable and then checked again.
//===----------------------------------------------------------------------===//
struct s_7114618 { int *p; };
void test_rdar_7114618(struct s_7114618 *s) {
  if (s->p) {
    int *p = s->p;
    if (!p) {
      // Infeasible
      int *dead = 0;
      *dead = 0xDEADBEEF; // no-warning
    }
  }
}

// Test pointers increment correctly.
void f() {
  int a[2];
  a[1] = 3;
  int *p = a;
  p++;
  if (*p != 3) {
    int *q = 0;
    *q = 3; // no-warning
  }
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7185607>
// Bit-fields of a struct should be invalidated when blasting the entire
// struct with an integer constant.
//===----------------------------------------------------------------------===//
struct test_7185607 {
  int x : 10;
  int y : 22;
};
int rdar_test_7185607() {
  struct test_7185607 s; // Uninitialized.
  *((unsigned *) &s) = 0U;
  return s.x; // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7242006> [RegionStore] compound literal assignment with
//  floats not honored
// This test case is mirrored in misc-ps.m, but this case is a negative.
//===----------------------------------------------------------------------===//
typedef float CGFloat;
typedef struct _NSSize {
    CGFloat width;
    CGFloat height;
} NSSize;

CGFloat rdar7242006_negative(CGFloat x) {
  NSSize y;
  return y.width; // expected-warning{{garbage}}
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7249340> - Allow binding of values to symbolic regions.
// This test case shows how RegionStore tracks the value bound to 'x'
// after the assignment.
//===----------------------------------------------------------------------===//
typedef int* ptr_rdar_7249340;
void rdar_7249340(ptr_rdar_7249340 x) {
  *x = 1;
  if (*x)
    return;
  int *p = 0;   // This is unreachable.
  *p = 0xDEADBEEF; // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7249327> - This test case tests both value tracking of
// array values and that we handle symbolic values that are casted
// between different integer types.  Note the assignment 'n = *a++'; here
// 'n' is and 'int' and '*a' is 'unsigned'.  Previously we got a false positive
// at 'x += *b++' (undefined value) because we got a false path.
//===----------------------------------------------------------------------===//
int rdar_7249327_aux(void);

void rdar_7249327(unsigned int A[2*32]) {
  int B[2*32];
  int *b;
  unsigned int *a;
  int x = 0;
  
  int n;
  
  a = A;
  b = B;
  
  n = *a++;
  if (n)
    *b++ = rdar_7249327_aux();

  a = A;
  b = B;
  
  n = *a++;
  if (n)
    x += *b++; // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/6914474> - Check that 'x' is invalidated because its
// address is passed in as a value to a struct.
//===----------------------------------------------------------------------===//
struct doodad_6914474 { int *v; };
extern void prod_6914474(struct doodad_6914474 *d);
int rdar_6914474(void) {
  int x;
  struct doodad_6914474 d;
  d.v = &x;
  prod_6914474(&d);
  return x; // no-warning
}

// Test invalidation of a single field.
struct s_test_field_invalidate {
  int x;
};
extern void test_invalidate_field(int *x);
int test_invalidate_field_test() {
  struct s_test_field_invalidate y;
  test_invalidate_field(&y.x);
  return y.x; // no-warning
}
int test_invalidate_field_test_positive() {
  struct s_test_field_invalidate y;
  return y.x; // expected-warning{{garbage}}
}

// This test case illustrates how a typeless array of bytes casted to a
// struct should be treated as initialized.  RemoveDeadBindings previously
// had a bug that caused 'x' to lose its default symbolic value after the
// assignment to 'p', thus causing 'p->z' to evaluate to "undefined".
struct ArrayWrapper { unsigned char y[16]; };
struct WrappedStruct { unsigned z; };

void test_handle_array_wrapper_helper();

int test_handle_array_wrapper() {
  struct ArrayWrapper x;
  test_handle_array_wrapper_helper(&x);
  struct WrappedStruct *p = (struct WrappedStruct*) x.y; // expected-warning{{Casting a non-structure type to a structure type and accessing a field can lead to memory access errors or data corruption}}
  return p->z;  // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7261075> [RegionStore] crash when 
//   handling load: '*((unsigned int *)"????")'
//===----------------------------------------------------------------------===//

int rdar_7261075(void) {
  unsigned int var = 0;
  if (var == *((unsigned int *)"????"))
    return 1;
  return 0;
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7275774> false path due to limited pointer 
//                          arithmetic constraints
//===----------------------------------------------------------------------===//

void rdar_7275774(void *data, unsigned n) {
  if (!(data || n == 0))
    return;
  
  unsigned short *p = (unsigned short*) data;
  unsigned short *q = p + (n / 2);

  if (p < q) {
    // If we reach here, 'p' cannot be null.  If 'p' is null, then 'n' must
    // be '0', meaning that this branch is not feasible.
    *p = *q; // no-warning
  }
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7312221>
//
//  Test that Objective-C instance variables aren't prematurely pruned
//  from the analysis state.
//===----------------------------------------------------------------------===//

struct rdar_7312221_value { int x; };

@interface RDar7312221
{
  struct rdar_7312221_value *y;
}
- (void) doSomething_7312221;
@end

extern struct rdar_7312221_value *rdar_7312221_helper();
extern int rdar_7312221_helper_2(id o);
extern void rdar_7312221_helper_3(int z);

@implementation RDar7312221
- (void) doSomething_7312221 {
  if (y == 0) {
    y = rdar_7312221_helper();
    if (y != 0) {
      y->x = rdar_7312221_helper_2(self);
      // The following use of 'y->x' previously triggered a null dereference, as the value of 'y'
      // before 'y = rdar_7312221_helper()' would be used.
      rdar_7312221_helper_3(y->x); // no-warning
    }
  }
}
@end

struct rdar_7312221_container {
  struct rdar_7312221_value *y;
};

extern int rdar_7312221_helper_4(struct rdar_7312221_container *s);

// This test case essentially matches the one in [RDar7312221 doSomething_7312221].
void doSomething_7312221_with_struct(struct rdar_7312221_container *Self) {
  if (Self->y == 0) {
    Self->y = rdar_7312221_helper();
    if (Self->y != 0) {
      Self->y->x = rdar_7312221_helper_4(Self);
      rdar_7312221_helper_3(Self->y->x); // no-warning
    }
  }
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7332673> - Just more tests cases for regions
//===----------------------------------------------------------------------===//

void rdar_7332673_test1() {
    char value[1];
    if ( *(value) != 1 ) {} // expected-warning{{The left operand of '!=' is a garbage value}}
}
int rdar_7332673_test2_aux(char *x);
void rdar_7332673_test2() {
    char *value;
    if ( rdar_7332673_test2_aux(value) != 1 ) {} // expected-warning{{1st function call argument is an uninitialized value}}
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7347252>: Because of a bug in
//   RegionStoreManager::RemoveDeadBindings(), the symbol for s->session->p
//   would incorrectly be pruned from the state after the call to
//   rdar7347252_malloc1(), and would incorrectly result in a warning about
//   passing a null pointer to rdar7347252_memcpy().
//===----------------------------------------------------------------------===//

struct rdar7347252_AA { char *p;};
typedef struct {
 struct rdar7347252_AA *session;
 int t;
 char *q;
} rdar7347252_SSL1;

int rdar7347252_f(rdar7347252_SSL1 *s);
char *rdar7347252_malloc1(int);
char *rdar7347252_memcpy1(char *d, char *s, int n) __attribute__((nonnull (1,2)));

int rdar7347252(rdar7347252_SSL1 *s) {
 rdar7347252_f(s);  // the SymbolicRegion of 's' is set a default binding of conjured symbol
 if (s->session->p == ((void*)0)) {
   if ((s->session->p = rdar7347252_malloc1(10)) == ((void*)0)) {
     return 0;
   }
   rdar7347252_memcpy1(s->session->p, "aa", 2); // no-warning
 }
 return 0;
}

//===----------------------------------------------------------------------===//
// PR 5316 - "crash when accessing field of lazy compound value"
//  Previously this caused a crash at the MemberExpr '.chr' when loading
//  a field value from a LazyCompoundVal
//===----------------------------------------------------------------------===//

typedef unsigned int pr5316_wint_t;
typedef pr5316_wint_t pr5316_REFRESH_CHAR;
typedef struct {
  pr5316_REFRESH_CHAR chr;
}
pr5316_REFRESH_ELEMENT;
static void pr5316(pr5316_REFRESH_ELEMENT *dst, const pr5316_REFRESH_ELEMENT *src) {
  while ((*dst++ = *src++).chr != L'\0')  ;
}

//===----------------------------------------------------------------------===//
// Exercise creating ElementRegion with symbolic super region.
//===----------------------------------------------------------------------===//
void element_region_with_symbolic_superregion(int* p) {
  int *x;
  int a;
  if (p[0] == 1)
    x = &a;
  if (p[0] == 1)
    (void)*x; // no-warning
}

//===----------------------------------------------------------------------===//
// Test returning an out-of-bounds pointer (CWE-466)
//===----------------------------------------------------------------------===//

static int test_cwe466_return_outofbounds_pointer_a[10];
int *test_cwe466_return_outofbounds_pointer() {
  int *p = test_cwe466_return_outofbounds_pointer_a+11;
  return p; // expected-warning{{Returned pointer value points outside the original object}}
}

//===----------------------------------------------------------------------===//
// PR 3135 - Test case that shows that a variable may get invalidated when its
// address is included in a structure that is passed-by-value to an unknown function.
//===----------------------------------------------------------------------===//

typedef struct { int *a; } pr3135_structure;
int pr3135_bar(pr3135_structure *x);
int pr3135() {
  int x;
  pr3135_structure y = { &x };
  // the call to pr3135_bar may initialize x
  if (pr3135_bar(&y) && x) // no-warning
    return 1;
  return 0;
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7403269> - Test that we handle compound initializers with
// partially unspecified array values. Previously this caused a crash.
//===----------------------------------------------------------------------===//

typedef struct RDar7403269 {
  unsigned x[10];
  unsigned y;
} RDar7403269;

void rdar7403269() {
  RDar7403269 z = { .y = 0 };
  if (z.x[4] == 0)
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning  
}

typedef struct RDar7403269_b {
  struct zorg { int w; int k; } x[10];
  unsigned y;
} RDar7403269_b;

void rdar7403269_b() {
  RDar7403269_b z = { .y = 0 };
  if (z.x[5].w == 0)
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}

void rdar7403269_b_pos() {
  RDar7403269_b z = { .y = 0 };
  if (z.x[5].w == 1)
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{Dereference of null pointer}}
}


//===----------------------------------------------------------------------===//
// Test that incrementing a non-null pointer results in a non-null pointer.
// (<rdar://problem/7191542>)
//===----------------------------------------------------------------------===//

void test_increment_nonnull_rdar_7191542(const char *path) {
  const char *alf = 0;
  
  for (;;) {
    // When using basic-store, we get a null dereference here because we lose information
    // about path after the pointer increment.
    char c = *path++; // no-warning
    if (c == 'a') {
      alf = path;
    }
    
    if (alf)
      return;
  }
}

//===----------------------------------------------------------------------===//
// Test that the store (implicitly) tracks values for doubles/floats that are
// uninitialized (<rdar://problem/6811085>)
//===----------------------------------------------------------------------===//

double rdar_6811085(void) {
  double u;
  return u + 10; // expected-warning{{The left operand of '+' is a garbage value}}
}

//===----------------------------------------------------------------------===//
// Path-sensitive tests for blocks.
//===----------------------------------------------------------------------===//

void indirect_block_call(void (^f)());

int blocks_1(int *p, int z) {
  __block int *q = 0;
  void (^bar)() = ^{ q = p; };
  
  if (z == 1) {
    // The call to 'bar' might cause 'q' to be invalidated.
    bar();
    *q = 0x1; // no-warning
  }
  else if (z == 2) {
    // The function 'indirect_block_call' might invoke bar, thus causing
    // 'q' to possibly be invalidated.
    indirect_block_call(bar);
    *q = 0x1; // no-warning
  }
  else {
    *q = 0xDEADBEEF; // expected-warning{{Dereference of null pointer}}
  }
  return z;
}

int blocks_2(int *p, int z) {
  int *q = 0;
  void (^bar)(int **) = ^(int **r){ *r = p; };
  
  if (z) {
    // The call to 'bar' might cause 'q' to be invalidated.
    bar(&q);
    *q = 0x1; // no-warning
  }
  else {
    *q = 0xDEADBEEF; // expected-warning{{Dereference of null pointer}}
  }
  return z;
}

// Test that the value of 'x' is considered invalidated after the block
// is passed as an argument to the message expression.
typedef void (^RDar7582031CB)(void);
@interface RDar7582031
- rdar7582031:RDar7582031CB;
- rdar7582031_b:RDar7582031CB;
@end

// Test with one block.
unsigned rdar7582031(RDar7582031 *o) {
  __block unsigned x;
  [o rdar7582031:^{ x = 1; }];
  return x; // no-warning
}

// Test with two blocks.
unsigned long rdar7582031_b(RDar7582031 *o) {
  __block unsigned y;
  __block unsigned long x;
  [o rdar7582031:^{ y = 1; }];
  [o rdar7582031_b:^{ x = 1LL; }];
  return x + (unsigned long) y; // no-warning
}

// Show we get an error when 'o' is null because the message
// expression has no effect.
unsigned long rdar7582031_b2(RDar7582031 *o) {
  __block unsigned y;
  __block unsigned long x;
  if (o)
    return 1;
  [o rdar7582031:^{ y = 1; }];
  [o rdar7582031_b:^{ x = 1LL; }];
  return x + (unsigned long) y; // expected-warning{{The left operand of '+' is a garbage value}}
}

// Show that we handle static variables also getting invalidated.
void rdar7582031_aux(void (^)(void));
RDar7582031 *rdar7582031_aux_2();

unsigned rdar7582031_static() {  
  static RDar7582031 *o = 0;
  rdar7582031_aux(^{ o = rdar7582031_aux_2(); });
  
  __block unsigned x;
  [o rdar7582031:^{ x = 1; }];
  return x; // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7462324> - Test that variables passed using __blocks
//  are not treated as being uninitialized.
//===----------------------------------------------------------------------===//

typedef void (^RDar_7462324_Callback)(id obj);

@interface RDar7462324
- (void) foo:(id)target;
- (void) foo_positive:(id)target;

@end

@implementation RDar7462324
- (void) foo:(id)target {
  __block RDar_7462324_Callback builder = ((void*) 0);
  builder = ^(id object) {
    if (object) {
      builder(self); // no-warning
    }
  };
  builder(target);
}
- (void) foo_positive:(id)target {
  __block RDar_7462324_Callback builder = ((void*) 0);
  builder = ^(id object) {
    id x;
    if (object) {
      builder(x); // expected-warning{{1st block call argument is an uninitialized value}}
    }
  };
  builder(target);
}
@end

//===----------------------------------------------------------------------===//
// <rdar://problem/7468209> - Scanning for live variables within a block should
//  not crash on variables passed by reference via __block.
//===----------------------------------------------------------------------===//

int rdar7468209_aux();
void rdar7468209_aux_2();

void rdar7468209() {
  __block int x = 0;
  ^{
    x = rdar7468209_aux();
    // We need a second statement so that 'x' would be removed from the store if it wasn't
    // passed by reference.
    rdar7468209_aux_2();
  }();
}

//===----------------------------------------------------------------------===//
// PR 5857 - Test loading an integer from a byte array that has also been
//  reinterpreted to be loaded as a field.
//===----------------------------------------------------------------------===//

typedef struct { int x; } TestFieldLoad;
int pr5857(char *src) {
  TestFieldLoad *tfl = (TestFieldLoad *) (intptr_t) src;
  int y = tfl->x;
  long long *z = (long long *) (intptr_t) src;
  long long w = 0;
  int n = 0;
  for (n = 0; n < y; ++n) {
    // Previously we crashed analyzing this statement.
    w = *z++;
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// PR 4358 - Without field-sensitivity, this code previously triggered
//  a false positive that 'uninit' could be uninitialized at the call
//  to pr4358_aux().
//===----------------------------------------------------------------------===//

struct pr4358 {
  int bar;
  int baz;
};
void pr4358_aux(int x);
void pr4358(struct pr4358 *pnt) {
  int uninit;
  if (pnt->bar < 3) {
    uninit = 1;
  } else if (pnt->baz > 2) {
    uninit = 3;
  } else if (pnt->baz <= 2) {
    uninit = 2;
  }
  pr4358_aux(uninit); // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7526777>
// Test handling fields of values returned from function calls or
// message expressions.
//===----------------------------------------------------------------------===//

typedef struct testReturn_rdar_7526777 {
  int x;
  int y;
} testReturn_rdar_7526777;

@interface TestReturnStruct_rdar_7526777
- (testReturn_rdar_7526777) foo;
@end

int test_return_struct(TestReturnStruct_rdar_7526777 *x) {
  return [x foo].x;
}

testReturn_rdar_7526777 test_return_struct_2_aux_rdar_7526777();

int test_return_struct_2_rdar_7526777() {
  return test_return_struct_2_aux_rdar_7526777().x;
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7527292> Assertion failed: (Op == BinaryOperator::Add || 
//                                             Op == BinaryOperator::Sub)
// This test case previously triggered an assertion failure due to a discrepancy
// been the loaded/stored value in the array
//===----------------------------------------------------------------------===//

_Bool OSAtomicCompareAndSwapPtrBarrier( void *__oldValue, void *__newValue, void * volatile *__theValue );

void rdar_7527292() {
  static id Cache7527292[32];
  for (signed long idx = 0;
       idx < 32;
       idx++) {
    id v = Cache7527292[idx];
    if (v && OSAtomicCompareAndSwapPtrBarrier(v, ((void*)0), (void * volatile *)(Cache7527292 + idx))) { 
    }
  }
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7515938> - Handle initialization of incomplete arrays
//  in structures using a compound value.  Previously this crashed.
//===----------------------------------------------------------------------===//

struct rdar_7515938 {
  int x;
  int y[];
};

const struct rdar_7515938 *rdar_7515938() {
  static const struct rdar_7515938 z = { 0, { 1, 2 } };
  if (z.y[0] != 1) {
    int *p = 0;
    *p = 0xDEADBEEF; // no-warning
  }
  return &z;
}

struct rdar_7515938_str {
  int x;
  char y[];
};

const struct rdar_7515938_str *rdar_7515938_str() {
  static const struct rdar_7515938_str z = { 0, "hello" };
  return &z;
}

//===----------------------------------------------------------------------===//
// Assorted test cases from PR 4172.
//===----------------------------------------------------------------------===//

struct PR4172A_s { int *a; };

void PR4172A_f2(struct PR4172A_s *p);

int PR4172A_f1(void) {
    struct PR4172A_s m;
    int b[4];
    m.a = b;
    PR4172A_f2(&m);
    return b[3]; // no-warning
}

struct PR4172B_s { int *a; };

void PR4172B_f2(struct PR4172B_s *p);

int PR4172B_f1(void) {
    struct PR4172B_s m;
    int x;
    m.a = &x;
    PR4172B_f2(&m);
    return x; // no-warning
}

//===----------------------------------------------------------------------===//
// Test invalidation of values in struct literals.
//===----------------------------------------------------------------------===//

struct s_rev96062 { int *x; int *y; };
struct s_rev96062_nested { struct s_rev96062 z; };

void test_a_rev96062_aux(struct s_rev96062 *s);
void test_a_rev96062_aux2(struct s_rev96062_nested *s);

int test_a_rev96062() {
  int a, b;
  struct s_rev96062 x = { &a, &b };
  test_a_rev96062_aux(&x);
  return a + b; // no-warning
}
int test_b_rev96062() {
  int a, b;
  struct s_rev96062 x = { &a, &b };
  struct s_rev96062 z = x;
  test_a_rev96062_aux(&z);
  return a + b; // no-warning
}
int test_c_rev96062() {
  int a, b;
  struct s_rev96062 x = { &a, &b };
  struct s_rev96062_nested w = { x };
  struct s_rev96062_nested z = w;
  test_a_rev96062_aux2(&z);
  return a + b; // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7242010> - The access to y[0] at the bottom previously
//  was reported as an uninitialized value.
//===----------------------------------------------------------------------===//

char *rdar_7242010(int count, char **y) {
  char **x = alloca((count + 4) * sizeof(*x));
  x[0] = "hi";
  x[1] = "there";
  x[2] = "every";
  x[3] = "body";
  memcpy(x + 4, y, count * sizeof(*x));
  y = x;
  return y[0]; // no-warning
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7770737>
//===----------------------------------------------------------------------===//

struct rdar_7770737_s { intptr_t p; };
void rdar_7770737_aux(struct rdar_7770737_s *p);
int rdar_7770737(void)
{ 
  int x;

  // Previously 'f' was not properly invalidated, causing the use of
  // an uninitailized value below.
  struct rdar_7770737_s f = { .p = (intptr_t)&x };
  rdar_7770737_aux(&f);
  return x; // no-warning
}
int rdar_7770737_pos(void)
{
  int x;
  struct rdar_7770737_s f = { .p = (intptr_t)&x };
  return x; // expected-warning{{Undefined or garbage value returned to caller}}
}

//===----------------------------------------------------------------------===//
// Test handling of the implicit 'isa' field.  For now we don't do anything
// interesting.
//===----------------------------------------------------------------------===//

void pr6302(id x, Class y) {
  // This previously crashed the analyzer (reported in PR 6302)
  x->isa  = y; // expected-warning {{assignment to Objective-C's isa is deprecated in favor of object_setClass()}}
}

//===----------------------------------------------------------------------===//
// Specially handle global variables that are declared constant.  In the
// example below, this forces the loop to take exactly 2 iterations.
//===----------------------------------------------------------------------===//

const int pr6288_L_N = 2;
void pr6288_(void) {
  int x[2];
  int *px[2];
  int i;
  for (i = 0; i < pr6288_L_N; i++)
    px[i] = &x[i];
  *(px[0]) = 0; // no-warning
}

void pr6288_pos(int z) {
  int x[2];
  int *px[2];
  int i;
  for (i = 0; i < z; i++)
    px[i] = &x[i]; // expected-warning{{Access out-of-bound array element (buffer overflow)}}
  *(px[0]) = 0; // expected-warning{{Dereference of undefined pointer value}}
}

void pr6288_b(void) {
  const int L_N = 2;
  int x[2];
  int *px[2];
  int i;
  for (i = 0; i < L_N; i++)
    px[i] = &x[i];
  *(px[0]) = 0; // no-warning
}

// <rdar://problem/7817800> - A bug in RemoveDeadBindings was causing instance variable bindings
//  to get prematurely pruned from the state.
@interface Rdar7817800 {
  char *x;
}
- (void) rdar7817800_baz;
@end

char *rdar7817800_foobar();
void rdar7817800_qux(void*);

@implementation Rdar7817800
- (void) rdar7817800_baz {
  if (x)
    rdar7817800_qux(x);
  x = rdar7817800_foobar();
  // Previously this triggered a bogus null dereference warning.
  x[1] = 'a'; // no-warning
}
@end

// PR 6036 - This test case triggered a crash inside StoreManager::CastRegion because the size
// of 'unsigned long (*)[0]' is 0.
struct pr6036_a { int pr6036_b; };
struct pr6036_c;
void u132monitk (struct pr6036_c *pr6036_d) {
  (void) ((struct pr6036_a *) (unsigned long (*)[0]) ((char *) pr6036_d - 1))->pr6036_b; // expected-warning{{Casting a non-structure type to a structure type and accessing a field can lead to memory access errors or data corruption}}
}

// <rdar://problem/7813989> - ?-expressions used as a base of a member expression should be treated as an lvalue
typedef struct rdar7813989_NestedVal { int w; } rdar7813989_NestedVal;
typedef struct rdar7813989_Val { rdar7813989_NestedVal nv; } rdar7813989_Val;

int rdar7813989(int x, rdar7813989_Val *a, rdar7813989_Val *b) {
  // This previously crashed with an assertion failure.
  int z = (x ? a->nv : b->nv).w;
  return z + 1;
}

// PR 6844 - Don't crash on vaarg expression.
typedef __builtin_va_list va_list;
void map(int srcID, ...) {
  va_list ap;
  int i;
  for (i = 0; i < srcID; i++) {
    int v = __builtin_va_arg(ap, int);
  }
}

// PR 6854 - crash when casting symbolic memory address to a float
// Handle casting from a symbolic region to a 'float'.  This isn't
// really all that intelligent, but previously this caused a crash
// in SimpleSValuator.
void pr6854(void * arg) {
  void * a = arg;
  *(void**)a = arg;
  float f = *(float*) a;
}

// <rdar://problem/8032791> False positive due to symbolic store not find
//  value because of 'const' qualifier
double rdar_8032791_2();
double rdar_8032791_1() {
   struct R8032791 { double x[2]; double y; }
   data[3] = {
     {{1.0, 3.0}, 3.0},  //  1   2   3
     {{1.0, 1.0}, 0.0},  // 1 1 2 2 3 3
     {{1.0, 3.0}, 1.0}   //    1   2   3
   };

   double x = 0.0;
   for (unsigned i = 0 ; i < 3; i++) {
     const struct R8032791 *p = &data[i];
     x += p->y + rdar_8032791_2(); // no-warning
   }
   return x;
}

// PR 7450 - Handle pointer arithmetic with __builtin_alloca
void pr_7450_aux(void *x);
void pr_7450() {
  void *p = __builtin_alloca(10);
  // Don't crash when analyzing the following statement.
  pr_7450_aux(p + 8);
}

// <rdar://problem/8243408> - Symbolicate struct values returned by value.
struct s_rdar_8243408 { int x; };
extern struct s_rdar_8243408 rdar_8243408_aux(void);
void rdar_8243408(void) {
  struct s_rdar_8243408 a = { 1 }, *b = 0;
  while (a.x && !b)
    a = rdar_8243408_aux();

  // Previously there was a false error here with 'b' being null.
  (void) (a.x && b->x); // no-warning

  // Introduce a null deref to ensure we are checking this path.
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{Dereference of null pointer}}
}

// <rdar://problem/8258814>
int r8258814()
{
  int foo;
  int * a = &foo;
  a[0] = 10;
  // Do not warn that the value of 'foo' is uninitialized.
  return foo; // no-warning
}

// PR 8052 - Don't crash when reasoning about loads from a function address.\n
typedef unsigned int __uint32_t;
typedef unsigned long vm_offset_t;
typedef __uint32_t pd_entry_t;
typedef unsigned char u_char;
typedef unsigned int u_int;
typedef unsigned long u_long;
extern int      bootMP_size;
void            bootMP(void);
static void 
pr8052(u_int boot_addr)
{
    int             x;
    int             size = *(int *) ((u_long) & bootMP_size);
    u_char         *src = (u_char *) ((u_long) bootMP);
    u_char         *dst = (u_char *) boot_addr + ((vm_offset_t) ((((((((1 <<
12) / (sizeof(pd_entry_t))) - 1) - 1) - (260 - 2))) << 22) | ((0) << 12)));
#ifdef TEST_64
// expected-warning@-3 {{cast to 'u_char *' (aka 'unsigned char *') from smaller integer type 'u_int' (aka 'unsigned int')}}
#endif
    for (x = 0;
         x < size;
         ++x)
        *dst++ = *src++;
}

// PR 8015 - don't return undefined values for arrays when using a valid
// symbolic index
int pr8015_A();
void pr8015_B(const char *);

void pr8015_C() {
  int number = pr8015_A();
  const char *numbers[] = { "zero" };    
  if (number == 0) {
      pr8015_B(numbers[number]); // no-warning
  }
}

// Tests that we correctly handle that 'number' is perfectly constrained
// after 'if (number == 0)', allowing us to resolve that
// numbers[number] == numbers[0].
void pr8015_D_FIXME() {
  int number = pr8015_A();
  const char *numbers[] = { "zero" };
  if (number == 0) {
    if (numbers[number] == numbers[0])
      return;
    // Unreachable.
    int *p = 0;
    *p = 0xDEADBEEF; // no-warnng
  }
}

void pr8015_E() {
  // Similar to pr8015_C, but number is allowed to be a valid range.
  unsigned number = pr8015_A();
  const char *numbers[] = { "zero", "one", "two" };
  if (number < 3) {
    pr8015_B(numbers[number]); // no-warning
  }
}

void pr8015_F_FIXME() {
  // Similar to pr8015_E, but like pr8015_D we check if the pointer
  // is the same as one of the string literals.  The null dereference
  // here is not feasible in practice, so this is a false positive.
  int number = pr8015_A();
  const char *numbers[] = { "zero", "one", "two" };
  if (number < 3) {
    const char *p = numbers[number];
    if (p == numbers[0] || p == numbers[1] || p == numbers[2])
      return;
    int *q = 0;
    *q = 0xDEADBEEF; // expected-warning{{Dereference of null pointer}}
  }
}

// PR 8141.  Previously the statement expression in the for loop caused
// the CFG builder to crash.
struct list_pr8141
{
  struct list_pr8141 *tail;
};

struct list_pr8141 *
pr8141 (void) {
  struct list_pr8141 *items;
  for (;; items = ({ do { } while (0); items->tail; })) // expected-warning{{dereference of an undefined pointer value}}
    {
    }
}

// Don't crash when building the CFG.
void do_not_crash(int x) {
  while (x - ({do {} while (0); x; })) {
  }
}

// <rdar://problem/8424269> - Handle looking at the size of a VLA in
// ArrayBoundChecker.  Nothing intelligent (yet); just don't crash.
typedef struct RDar8424269_A {
  int RDar8424269_C;
} RDar8424269_A;
static void RDar8424269_B(RDar8424269_A *p, unsigned char *RDar8424269_D,
                          const unsigned char *RDar8424269_E, int RDar8424269_F,
    int b_w, int b_h, int dx, int dy) {
  int x, y, b, r, l;
  unsigned char tmp2t[3][RDar8424269_F * (32 + 8)];
  unsigned char *tmp2 = tmp2t[0];
  if (p && !p->RDar8424269_C)
    b = 15;
  tmp2 = tmp2t[1];
  if (b & 2) { // expected-warning{{The left operand of '&' is a garbage value}}
    for (y = 0; y < b_h; y++) {
      for (x = 0; x < b_w + 1; x++) {
        int am = 0;
        tmp2[x] = am;
      }
    }
  }
  tmp2 = tmp2t[2];
}

// <rdar://problem/8642434> - Handle transparent unions with the NonNullParamChecker.
typedef union {
  struct rdar_8642434_typeA *_dq;
}
rdar_8642434_typeB __attribute__((transparent_union));

__attribute__((visibility("default"))) __attribute__((__nonnull__)) __attribute__((__nothrow__))
void rdar_8642434_funcA(rdar_8642434_typeB object);

void rdar_8642434_funcB(struct rdar_8642434_typeA *x, struct rdar_8642434_typeA *y) {
  rdar_8642434_funcA(x);
  if (!y)
    rdar_8642434_funcA(y); // expected-warning{{Null pointer passed to 1st parameter expecting 'nonnull'}}
}

// <rdar://problem/8848957> - Handle loads and stores from a symbolic index
// into array without warning about an uninitialized value being returned.
// While RegionStore can't fully reason about this example, it shouldn't
// warn here either.
typedef struct s_test_rdar8848957 {
  int x, y, z;
} s_test_rdar8848957;

s_test_rdar8848957 foo_rdar8848957();
int rdar8848957(int index) {
  s_test_rdar8848957 vals[10];
  vals[index] = foo_rdar8848957();
  return vals[index].x; // no-warning
}

// PR 9049 - crash on symbolicating unions.  This test exists solely to
// test that the analyzer doesn't crash.
typedef struct pr9048_cdev *pr9048_cdev_t;
typedef union pr9048_abstracted_disklabel { void *opaque; } pr9048_disklabel_t;
struct pr9048_diskslice { pr9048_disklabel_t ds_label; };
struct pr9048_diskslices {
  int dss_secmult;
  struct pr9048_diskslice dss_slices[16];
};
void pr9048(pr9048_cdev_t dev, struct pr9048_diskslices * ssp, unsigned int slice)
{
  pr9048_disklabel_t     lp;
  struct pr9048_diskslice *sp;
  sp = &ssp->dss_slices[slice];
  if (ssp->dss_secmult == 1) {
  } else if ((lp = sp->ds_label).opaque != ((void *) 0)) {
  }
}

// Test Store reference counting in the presence of Lazy compound values.
// This previously caused an infinite recursion.
typedef struct {} Rdar_9103310_A;
typedef struct Rdar_9103310_B Rdar_9103310_B_t;
struct Rdar_9103310_B {
  unsigned char           Rdar_9103310_C[101];
};
void Rdar_9103310_E(Rdar_9103310_A * x, struct Rdar_9103310_C * b) { // expected-warning {{declaration of 'struct Rdar_9103310_C' will not be visible outside of this function}}
  char Rdar_9103310_D[4][4] = { "a", "b", "c", "d"};
  int i;
  Rdar_9103310_B_t *y = (Rdar_9103310_B_t *) x;
  for (i = 0; i < 101; i++) {
    Rdar_9103310_F(b, "%2d%s ", (y->Rdar_9103310_C[i]) / 4, Rdar_9103310_D[(y->Rdar_9103310_C[i]) % 4]); // expected-warning {{implicit declaration of function 'Rdar_9103310_F' is invalid in C99}}
  }
}

// Test handling binding lazy compound values to a region and then have
// specific elements have other bindings.
int PR9455() {
  char arr[4] = "000";
  arr[0] = '1';
  if (arr[1] == '0')
    return 1;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
  return 1;
}
int PR9455_2() {
  char arr[4] = "000";
  arr[0] = '1';
  if (arr[1] == '0') {
    int *p = 0;
    *p = 0xDEADBEEF; // expected-warning {{null}}
  }
  return 1;
}

// Test initialization of substructs via lazy compound values.
typedef float RDar9163742_Float;

typedef struct {
    RDar9163742_Float x, y;
} RDar9163742_Point;
typedef struct {
    RDar9163742_Float width, height;
} RDar9163742_Size;
typedef struct {
    RDar9163742_Point origin;
    RDar9163742_Size size;
} RDar9163742_Rect;

extern  RDar9163742_Rect RDar9163742_RectIntegral(RDar9163742_Rect);

RDar9163742_Rect RDar9163742_IntegralRect(RDar9163742_Rect frame)
{
    RDar9163742_Rect integralFrame;
    integralFrame.origin.x = frame.origin.x;
    integralFrame.origin.y = frame.origin.y;
    integralFrame.size = frame.size;
    return RDar9163742_RectIntegral(integralFrame); // no-warning; all fields initialized
}

// Test correct handling of prefix '--' operator.
void rdar9444714() {
  int   x;
  char    str[ 32 ];
  char    buf[ 32 ];
  char *  dst;
  char *  ptr;

  x = 1234;
  dst = str;
  ptr = buf;
  do
  {
    *ptr++ = (char)( '0' + ( x % 10 ) );
    x /= 10;  
  } while( x > 0 );

  while( ptr > buf )
  {
    *dst++ = *( --( ptr ) ); // no-warning
  }
  *dst = '\0';
}

// Test handling symbolic elements with field accesses.
// <rdar://problem/11127008>
typedef struct {
    unsigned value;
} RDar11127008;

signed rdar_11127008_index();

static unsigned rdar_11127008(void) {
    RDar11127008 values[] = {{.value = 0}, {.value = 1}};
    signed index = rdar_11127008_index();
    if (index < 0) return 0;
    if (index >= 2) return 0;
    return values[index].value;
}

// Test handling invalidating arrays passed to a block via captured
// pointer value (not a __block variable).
typedef void (^radar11125868_cb)(int *, unsigned);

void rdar11125868_aux(radar11125868_cb cb);

int rdar11125868() {
  int integersStackArray[1];
  int *integers = integersStackArray;
  rdar11125868_aux(^(int *integerValue, unsigned index) {
      integers[index] = integerValue[index];
    });
  return integers[0] == 0; // no-warning
}

int rdar11125868_positive() {
  int integersStackArray[1];
  int *integers = integersStackArray;
  return integers[0] == 0; // expected-warning {{The left operand of '==' is a}}
}
