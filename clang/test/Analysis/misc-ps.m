// NOTE: Use '-fobjc-gc' to test the analysis being run twice, and multiple reports are not issued.
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -analyzer-checker=core,alpha.core,osx.cocoa.AtSync -Wno-strict-prototypes -Wno-pointer-to-int-cast -verify -fblocks -Wno-unreachable-code -Wno-null-dereference -Wno-objc-root-class %s
// RUN: %clang_analyze_cc1 -triple x86_64-apple-darwin10 -analyzer-checker=core,alpha.core,osx.cocoa.AtSync -Wno-strict-prototypes -Wno-pointer-to-int-cast -verify -fblocks -Wno-unreachable-code -Wno-null-dereference -Wno-objc-root-class %s

#ifndef __clang_analyzer__
#error __clang_analyzer__ not defined
#endif

typedef struct objc_ivar *Ivar;
typedef struct objc_selector *SEL;
typedef signed char BOOL;
typedef int NSInteger;
typedef unsigned int NSUInteger;
typedef struct _NSZone NSZone;
@class NSInvocation, NSArray, NSMethodSignature, NSCoder, NSString, NSEnumerator;
@protocol NSObject
- (BOOL)isEqual:(id)object;
- (id)autorelease;
@end
@protocol NSCopying
- (id)copyWithZone:(NSZone *)zone;
@end
@protocol NSMutableCopying  - (id)mutableCopyWithZone:(NSZone *)zone; @end
@protocol NSCoding
- (void)encodeWithCoder:(NSCoder *)aCoder;
@end
@interface NSObject <NSObject> {}
- (id)init;
+ (id)allocWithZone:(NSZone *)zone;
@end
extern id NSAllocateObject(Class aClass, NSUInteger extraBytes, NSZone *zone);
@interface NSString : NSObject <NSCopying, NSMutableCopying, NSCoding>
- (NSUInteger)length;
+ (id)stringWithUTF8String:(const char *)nullTerminatedCString;
@end extern NSString * const NSBundleDidLoadNotification;
@interface NSValue : NSObject <NSCopying, NSCoding>
- (void)getValue:(void *)value;
@end
@interface NSNumber : NSValue
- (char)charValue;
- (id)initWithBool:(BOOL)value;
@end
@interface NSAssertionHandler : NSObject {}
+ (NSAssertionHandler *)currentHandler;
- (void)handleFailureInMethod:(SEL)selector object:(id)object file:(NSString *)fileName lineNumber:(NSInteger)line description:(NSString *)format,...;
@end
extern NSString * const NSConnectionReplyMode;
typedef float CGFloat;
typedef struct _NSPoint {
    CGFloat x;
    CGFloat y;
} NSPoint;
typedef struct _NSSize {
    CGFloat width;
    CGFloat height;
} NSSize;
typedef struct _NSRect {
    NSPoint origin;
    NSSize size;
} NSRect;

// Reduced test case from crash in <rdar://problem/6253157>
@interface A @end
@implementation A
- (void)foo:(void (^)(NSObject *x))block {
  if (!((block != ((void *)0)))) {}
}
@end

// Reduced test case from crash in PR 2796;
//  http://llvm.org/bugs/show_bug.cgi?id=2796

unsigned foo(unsigned x) { return __alignof__((x)) + sizeof(x); }

// Improvement to path-sensitivity involving compound assignments.
//  Addresses false positive in <rdar://problem/6268365>
//

unsigned r6268365Aux(void);

void r6268365(void) {
  unsigned x = 0;
  x &= r6268365Aux();
  unsigned j = 0;
    
  if (x == 0) ++j;
  if (x == 0) x = x / j;
}

void divzeroassume(unsigned x, unsigned j) {  
  x /= j;  
  if (j == 0) x /= 0;     // no static-analyzer warning    expected-warning {{division by zero is undefined}}
  if (j == 0) x /= j;     // no static-analyzer warning
  if (j == 0) x = x / 0;  // no static-analyzer warning    expected-warning {{division by zero is undefined}}
}

void divzeroassumeB(unsigned x, unsigned j) {  
  x = x / j;  
  if (j == 0) x /= 0;     // no static-analyzer warning     expected-warning {{division by zero is undefined}}
  if (j == 0) x /= j;     // no static-analyzer warning
  if (j == 0) x = x / 0;  // no static-analyzer warning     expected-warning {{division by zero is undefined}}
}

// InitListExpr processing

typedef float __m128 __attribute__((__vector_size__(16), __may_alias__));
__m128 return128(void) {
  // This compound literal has a Vector type.  We currently just
  // return UnknownVal.
  return __extension__(__m128) { 0.0f, 0.0f, 0.0f, 0.0f };
}

typedef long long __v2di __attribute__ ((__vector_size__ (16)));
typedef long long __m128i __attribute__ ((__vector_size__ (16), __may_alias__));
__m128i vec128i(long long __q1, long long __q0) {
  // This compound literal returns true for both isVectorType() and 
  // isIntegerType().
  return __extension__ (__m128i)(__v2di){ __q0, __q1 };
}

// sizeof(void)
// - Tests a regression reported in PR 3211: http://llvm.org/bugs/show_bug.cgi?id=3211
void handle_sizeof_void(unsigned flag) {
  int* p = 0;

  if (flag) {
    if (sizeof(void) == 1)
      return;
    // Infeasible.
    *p = 1; // no-warning
  }
  
  void* q;
  
  if (!flag) {
    if (sizeof(*q) == 1) // expected-warning {{ISO C does not allow indirection on operand of type 'void *'}}
      return;
    // Infeasibe.
    *p = 1; // no-warning
  }
    
  // Infeasible.
  *p = 1; // no-warning
}

// check deference of undefined values
void check_deref_undef(void) {
  int *p;
  *p = 0xDEADBEEF; // expected-warning{{Dereference of undefined pointer value}}
}

// PR 3422
void pr3422_helper(char *p);
void pr3422(void) {
  char buf[100];
  char *q = &buf[10];
  pr3422_helper(&q[1]);
}

// PR 3543 (handle empty statement expressions)
void pr_3543(void) {
  ({});
}

// <rdar://problem/6611677>
// This test case test the use of a vector type within an array subscript
// expression.
typedef long long __a64vector __attribute__((__vector_size__(8)));
typedef long long __a128vector __attribute__((__vector_size__(16)));
static inline __a64vector __attribute__((__always_inline__, __nodebug__))  
my_test_mm_movepi64_pi64(__a128vector a) {
  return (__a64vector)a[0];
}

// Test basic tracking of ivars associated with 'self'.
@interface SelfIvarTest : NSObject {
  int flag;
}
- (void)test_self_tracking;
@end

@implementation SelfIvarTest
- (void)test_self_tracking {
  char *p = 0;
  char c;

  if (flag)
    p = "hello";

  if (flag)
    c = *p; // no-warning
}
@end

// PR 3770
char pr3770(int x) {
  int y = x & 0x2;
  char *p = 0;
  if (y == 1)
    p = "hello";

  if (y == 1)
    return p[0]; // no-warning
    
  return 'a';
}

// PR 3772
// - We just want to test that this doesn't crash the analyzer.
typedef struct st ST;
struct st { char *name; };
extern ST *Cur_Pu;

void pr3772(void)
{
  static ST *last_Cur_Pu;
  if (last_Cur_Pu == Cur_Pu) {
    return;
  } 
}

// PR 3780 - This tests that StmtIterator isn't broken for VLAs in DeclGroups.
void pr3780(int sz) { typedef double MAT[sz][sz]; }

// <rdar://problem/6695527> - Test that we don't symbolicate doubles before
// we are ready to do something with them.
int rdar6695527(double x) {
  if (!x) { return 0; }
  return 1;
}

// <rdar://problem/6708148> - Test that we properly invalidate structs
//  passed-by-reference to a function.
void pr6708148_invalidate(NSRect *x);
void pr6708148_use(NSRect x);
void pr6708148_test(void) {
  NSRect x;
  pr6708148_invalidate(&x);
  pr6708148_use(x); // no-warning
}

// Handle both kinds of noreturn attributes for pruning paths.
void rdar_6777003_noret(void) __attribute__((noreturn));
void rdar_6777003_analyzer_noret(void) __attribute__((analyzer_noreturn));

void rdar_6777003(int x) {
  int *p = 0;
  
  if (x == 1) {
    rdar_6777003_noret();
    *p = 1; // no-warning;    
  }
  
  if (x == 2) {
    rdar_6777003_analyzer_noret();
    *p = 1; // no-warning;
  }
  
  *p = 1; // expected-warning{{Dereference of null pointer}}  
}

// Check that the pointer-to-conts arguments do not get invalidated by Obj C 
// interfaces. radar://10595327
int rdar_10595327(char *str) {
  char fl = str[0]; 
  int *p = 0;
  NSString *s = [NSString stringWithUTF8String:str];
  if (str[0] != fl)
      return *p; // no-warning
  return 0;
}

// For pointer arithmetic, --/++ should be treated as preserving non-nullness,
// regardless of how well the underlying StoreManager reasons about pointer
// arithmetic.
// <rdar://problem/6777209>
void rdar_6777209(char *p) {
  if (p == 0)
    return;
  
  ++p;
  
  // This branch should always be infeasible.
  if (p == 0)
    *p = 'c'; // no-warning
}

// PR 4033.  A symbolic 'void *' pointer can be used as the address for a
// computed goto.
typedef void *Opcode;
Opcode pr_4033_getOpcode(void);
void pr_4033(void) {
  void *lbl = &&next_opcode;
next_opcode:
  {
    Opcode op = pr_4033_getOpcode();
    if (op) goto *op;
  }
}

// Test invalidating pointers-to-pointers with slightly different types.  This
// example came from a recent false positive due to a regression where the
// branch condition was falsely reported as being uninitialized.
void invalidate_by_ref(char **x);
int test_invalidate_by_ref(void) {
  unsigned short y;
  invalidate_by_ref((char**) &y);
  if (y) // no-warning
    return 1;
  return 0;  
}

// Test for <rdar://problem/7027684>.  This just tests that the CFG is
// constructed correctly.  Previously, the successor block of the entrance
// was the block containing the merge for '?', which would trigger an
// assertion failure.
int rdar_7027684_aux(void);
int rdar_7027684_aux_2(void) __attribute__((noreturn));
void rdar_7027684(int x, int y) {
  {}; // this empty compound statement is critical.
  (rdar_7027684_aux() ? rdar_7027684_aux_2() : (void) 0);
}

// Test that we handle casts of string literals to arbitrary types.
unsigned const char *string_literal_test1(void) {
  return (const unsigned char*) "hello";
}

const float *string_literal_test2(void) {
  return (const float*) "hello";
}

// Test that we handle casts *from* incomplete struct types.
extern const struct _FooAssertStruct _cmd;
void test_cast_from_incomplete_struct_aux(volatile const void *x);
void test_cast_from_incomplete_struct(void) {
  test_cast_from_incomplete_struct_aux(&_cmd);
}

// Test for <rdar://problem/7034511> 
//  "ValueManager::makeIntVal(uint64_t X, QualType T) should return a 'Loc' 
//   when 'T' is a pointer"
//
// Previously this case would crash.
void test_rdar_7034511(NSArray *y) {
  NSObject *x;
  for (x in y) {}
  if (x == ((void*) 0)) {}
}

// Handle casts of function pointers (CodeTextRegions) to arbitrary pointer
// types. This was previously causing a crash in CastRegion.
void handle_funcptr_voidptr_casts(void) {
  void **ptr;
  typedef void *PVOID;
  typedef void *PCHAR;  
  typedef long INT_PTR, *PINT_PTR;
  typedef INT_PTR (*FARPROC)(void);
  FARPROC handle_funcptr_voidptr_casts_aux(void);
  PVOID handle_funcptr_voidptr_casts_aux_2(PVOID volatile *x);
  PVOID handle_funcptr_voidptr_casts_aux_3(PCHAR volatile *x);  
  
  ptr = (void**) handle_funcptr_voidptr_casts_aux();
  handle_funcptr_voidptr_casts_aux_2(ptr);
  handle_funcptr_voidptr_casts_aux_3(ptr);
}

// RegionStore::Retrieve previously crashed on this example.  This example
// was previously in the test file 'xfail_regionstore_wine_crash.c'.
void testA(void) {
  long x = 0;
  char *y = (char *) &x;
  if (!*y)
    return;
}

// RegionStoreManager previously crashed on this example.  The problem is that
// the value bound to the field of b->grue after the call to testB_aux is
// a symbolic region.  The second '*__gruep__' involves performing a load
// from a 'int*' that really is a 'void**'.  The loaded location must be
// implicitly converted to an integer that wraps a location.  Previosly we would
// get a crash here due to an assertion failure.
typedef struct _BStruct { void *grue; } BStruct;
void testB_aux(void *ptr);
void testB(BStruct *b) {
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    testB_aux(__gruep__);
  }
  {
    int *__gruep__ = ((int *)&((b)->grue));
    int __gruev__ = *__gruep__;
    if (~0 != __gruev__) {}
  }
}

void test_trivial_symbolic_comparison(int *x) {
  int test_trivial_symbolic_comparison_aux(void);
  int a = test_trivial_symbolic_comparison_aux();
  int b = a;
  if (a != b) {
    int *p = 0;
    *p = 0xDEADBEEF;     // no-warning
  }
  
  a = a == 1;
  b = b == 1;
  if (a != b) {
    int *p = 0;
    *p = 0xDEADBEEF;     // no-warning
  }
}

// Test for:
//  <rdar://problem/7062158> false positive null dereference due to
//   BasicStoreManager not tracking *static* globals
//
// This just tests the proper tracking of symbolic values for globals (both 
// static and non-static).
//
static int* x_rdar_7062158;
void rdar_7062158(void) {
  int *current = x_rdar_7062158;
  if (current == x_rdar_7062158)
    return;
    
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning  
}

int* x_rdar_7062158_2;
void rdar_7062158_2(void) {
  int *current = x_rdar_7062158_2;
  if (current == x_rdar_7062158_2)
    return;
    
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning  
}

// This test reproduces a case for a crash when analyzing ClamAV using
// RegionStoreManager (the crash doesn't exhibit in BasicStoreManager because
// it isn't doing anything smart about arrays).  The problem is that on the
// second line, 'p = &p[i]', p is assigned an ElementRegion whose index
// is a 16-bit integer.  On the third line, a new ElementRegion is created
// based on the previous region, but there the region uses a 32-bit integer,
// resulting in a clash of values (an assertion failure at best).  We resolve
// this problem by implicitly converting index values to 'int' when the
// ElementRegion is created.
unsigned char test_array_index_bitwidth(const unsigned char *p) {
  unsigned short i = 0;
  for (i = 0; i < 2; i++) p = &p[i];
  return p[i+1];
}

// This case tests that CastRegion handles casts involving BlockPointerTypes.
// It should not crash.
void test_block_cast(void) {
  id test_block_cast_aux(void);
  (void (^)(void *))test_block_cast_aux(); // expected-warning{{expression result unused}}
}

int OSAtomicCompareAndSwap32Barrier();

// Test comparison of 'id' instance variable to a null void* constant after
// performing an OSAtomicCompareAndSwap32Barrier.
// This previously was a crash in RegionStoreManager.
@interface TestIdNull {
  id x;
}
-(int)foo;
@end
@implementation TestIdNull
-(int)foo {
  OSAtomicCompareAndSwap32Barrier(0, (signed)2, (signed*)&x);  
  if (x == (void*) 0) { return 0; }
  return 1;
}
@end

// Do not crash when performing compare and swap on symbolic values.
typedef int int32_t;
typedef int int32;
typedef int32 Atomic32;
int OSAtomicCompareAndSwap32( int32_t __oldValue, int32_t __newValue, volatile int32_t *__theValue);
void radar11390991_NoBarrier_CompareAndSwap(volatile Atomic32 *ptr,
                              Atomic32 old_value,
                              Atomic32 new_value) {
  OSAtomicCompareAndSwap32(old_value, new_value, ptr);
}

// PR 4594 - This was a crash when handling casts in SimpleSValuator.
void PR4594(void) {
  char *buf[1];
  char **foo = buf;
  *foo = "test";
}

// Test invalidation logic where an integer is casted to an array with a
// different sign and then invalidated.
void test_invalidate_cast_int(void) {
  void test_invalidate_cast_int_aux(unsigned *i);
  signed i;  
  test_invalidate_cast_int_aux((unsigned*) &i);
  if (i < 0)
    return;
}

int ivar_getOffset();

// Reduced from a crash involving the cast of an Objective-C symbolic region to
// 'char *'
static NSNumber *test_ivar_offset(id self, SEL _cmd, Ivar inIvar) {
  return [[[NSNumber allocWithZone:((void*)0)] initWithBool:*(_Bool *)((char *)self + ivar_getOffset(inIvar))] autorelease];
}

// Reduced from a crash in StoreManager::CastRegion involving a divide-by-zero.
// This resulted from not properly handling region casts to 'const void*'.
void test_cast_const_voidptr(void) {
  char x[10];
  char *p = &x[1];
  const void* q = p;
}

// Reduced from a crash when analyzing Wine.  This test handles loads from
// function addresses.
typedef long (*FARPROC)(void);
FARPROC test_load_func(FARPROC origfun) {
  if (!*(unsigned char*) origfun)
    return origfun;
  return 0;
}

// Test passing-by-value an initialized struct variable.
struct test_pass_val {
  int x;
  int y;
};
void test_pass_val_aux(struct test_pass_val s);
void test_pass_val(void) {
  struct test_pass_val s;
  s.x = 1;
  s.y = 2;
  test_pass_val_aux(s);
}

// This is a reduced test case of a false positive that previously appeared
// in RegionStoreManager.  Previously the array access resulted in dereferencing
// an undefined value.
int test_array_compound(int *q, int *r, int *z) {
  int *array[] = { q, r, z };
  int j = 0;
  for (unsigned i = 0; i < 3 ; ++i)
    if (*array[i]) ++j; // no-warning
  return j;
}

// symbolic value stored in 'x' wouldn't be implicitly casted to a signed value
// during the comparison.
int rdar_7124210(unsigned int x) {
  enum { SOME_CONSTANT = 123 };
  int compare = ((signed) SOME_CONSTANT) == *((signed *) &x);
  return compare ? 0 : 1; // Forces the evaluation of the symbolic constraint.
}

void pr4781(unsigned long *raw1) {
  unsigned long *cook, *raw0;
  unsigned long dough[32];
  int i;
  cook = dough;
  for( i = 0; i < 16; i++, raw1++ ) {
    raw0 = raw1++;
    *cook = (*raw0 & 0x00fc0000L) << 6;
    *cook |= (*raw0 & 0x00000fc0L) << 10;
  }
}

// <rdar://problem/7185647> - 'self' should be treated as being non-null
// upon entry to an objective-c method.
@interface RDar7185647
- (id)foo;
@end
@implementation RDar7185647
- (id) foo {
  if (self)
    return self;
  *((volatile int *) 0x0) = 0xDEADBEEF; // no-warning
  return self;
}
@end

// Test reasoning of __builtin_offsetof;
struct test_offsetof_A {
  int x;
  int y;
};
struct test_offsetof_B {
  int w;
  int z;
};
void test_offsetof_1(void) {
  if (__builtin_offsetof(struct test_offsetof_A, x) ==
      __builtin_offsetof(struct test_offsetof_B, w))
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}
void test_offsetof_2(void) {
  if (__builtin_offsetof(struct test_offsetof_A, y) ==
      __builtin_offsetof(struct test_offsetof_B, z))
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}
void test_offsetof_3(void) {
  if (__builtin_offsetof(struct test_offsetof_A, y) -
      __builtin_offsetof(struct test_offsetof_A, x)
      ==
      __builtin_offsetof(struct test_offsetof_B, z) -
      __builtin_offsetof(struct test_offsetof_B, w))
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}
void test_offsetof_4(void) {
  if (__builtin_offsetof(struct test_offsetof_A, y) ==
      __builtin_offsetof(struct test_offsetof_B, w))
    return;
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{Dereference of null pointer}}
}

// <rdar://problem/6829164> "nil receiver" false positive: make tracking 
// of the MemRegion for 'self' path-sensitive
@interface RDar6829164 : NSObject {
  double x; int y;
}
- (id) init;
@end

id rdar_6829164_1(void);
double rdar_6829164_2(void);

@implementation RDar6829164
- (id) init {
  if((self = [super init]) != 0) {
    id z = rdar_6829164_1();
    y = (z != 0);
    if (y)
      x = rdar_6829164_2();
  }
  return self;
}
@end

// <rdar://problem/7242015> - Invalidate values passed-by-reference
// to functions when the pointer to the value is passed as an integer.
void test_7242015_aux(unsigned long);
int rdar_7242015(void) {
  int x;
  test_7242015_aux((unsigned long) &x); // no-warning
  return x; // Previously we return and uninitialized value when
            // using RegionStore.
}

// <rdar://problem/7242006> [RegionStore] compound literal assignment with
//  floats not honored
CGFloat rdar7242006(CGFloat x) {
  NSSize y = (NSSize){x, 10};
  return y.width; // no-warning
}

// PR 4988 - This test exhibits a case where a function can be referenced
//  when not explicitly used in an "lvalue" context (as far as the analyzer is
//  concerned). This previously triggered a crash due to an invalid assertion.
void pr_4988(void) {
  pr_4988; // expected-warning{{expression result unused}}
}

// <rdar://problem/7152418> - A 'signed char' is used as a flag, which is
//  implicitly converted to an int.
void *rdar7152418_bar(void);
@interface RDar7152418 {
  signed char x;
}
-(char)foo;
@end;
@implementation RDar7152418
-(char)foo {
  char *p = 0;
  void *result = 0;
  if (x) {
    result = rdar7152418_bar();
    p = "hello";
  }
  if (!result) {
    result = rdar7152418_bar();
    if (result && x)
      return *p; // no-warning
  }
  return 1;
}

//===----------------------------------------------------------------------===//
// Test constant-folding of symbolic values, automatically handling type
// conversions of the symbol as necessary.
//===----------------------------------------------------------------------===//

// Previously this would crash once we started eagerly evaluating symbols whose 
// values were constrained to a single value.
void test_symbol_fold_1(signed char x) {
  while (1) {
    if (x == ((signed char) 0)) {}
  }
}

// This previously caused a crash because it triggered an assertion in APSInt.
void test_symbol_fold_2(unsigned int * p, unsigned int n,
                        const unsigned int * grumpkin, unsigned int dn) {
  unsigned int i;
  unsigned int tempsub[8];
  unsigned int *solgrumpkin = tempsub + n;
  for (i = 0; i < n; i++)
    solgrumpkin[i] = (i < dn) ? ~grumpkin[i] : 0xFFFFFFFF;
  for (i <<= 5; i < (n << 5); i++) {}
}

// This previously caused a crash because it triggered an assertion in APSInt.
// 'x' would evaluate to a 8-bit constant (because of the return value of
// test_symbol_fold_3_aux()) which would not get properly promoted to an
// integer.
char test_symbol_fold_3_aux(void);
unsigned test_symbol_fold_3(void) {
  unsigned x = test_symbol_fold_3_aux();
  if (x == 54)
    return (x << 8) | 0x5;
  return 0;
} 

//===----------------------------------------------------------------------===//
// Tests for the warning of casting a non-struct type to a struct type
//===----------------------------------------------------------------------===//

typedef struct {unsigned int v;} NSSwappedFloat;

NSSwappedFloat test_cast_nonstruct_to_struct(float x) {
  struct hodor {
    float number;
    NSSwappedFloat sf;
  };
  return ((struct hodor *)&x)->sf; // expected-warning{{Casting a non-structure type to a structure type and accessing a field can lead to memory access errors or data corruption}}
}

NSSwappedFloat test_cast_nonstruct_to_union(float x) {
  union bran {
    float number;
    NSSwappedFloat sf;
  };
  return ((union bran *)&x)->sf; // no-warning
}

void test_undefined_array_subscript(void) {
  int i, a[10];
  int *p = &a[i]; // expected-warning{{Array subscript is undefined}}
}
@end

//===----------------------------------------------------------------------===//
// Test using an uninitialized value as a branch condition.
//===----------------------------------------------------------------------===//

int test_uninit_branch(void) {
  int x;
  if (x) // expected-warning{{Branch condition evaluates to a garbage value}}
    return 1;
  return 0; 
}

int test_uninit_branch_b(void) {
  int x;
  return x ? 1 : 0; // expected-warning{{Branch condition evaluates to a garbage value}}
}

int test_uninit_branch_c(void) {
  int x;
  if ((short)x) // expected-warning{{Branch condition evaluates to a garbage value}}
    return 1;
  return 0; 
}

//===----------------------------------------------------------------------===//
// Test passing an undefined value in a message or function call.
//===----------------------------------------------------------------------===//

void test_bad_call_aux(int x);
void test_bad_call(void) {
  int y;
  test_bad_call_aux(y); // expected-warning{{1st function call argument is an uninitialized value}}
}

@interface TestBadArg {}
- (void) testBadArg:(int) x;
@end

void test_bad_msg(TestBadArg *p) {
  int y;
  [p testBadArg:y]; // expected-warning{{1st argument in message expression is an uninitialized value}}
}

//===----------------------------------------------------------------------===//
// PR 6033 - Test emitting the correct output in a warning where we use '%'
//  with operands that are undefined.
//===----------------------------------------------------------------------===//

int pr6033(int x) {
  int y;
  return x % y; // expected-warning{{The right operand of '%' is a garbage value}}
}

struct trie {
  struct trie* next;
};

struct kwset {
  struct trie *trie;
  unsigned char y[10];
  struct trie* next[10];
  int d;
};

typedef struct trie trie_t;
typedef struct kwset kwset_t;

void f(kwset_t *kws, char const *p, char const *q) {
  struct trie const *trie;
  struct trie * const *next = kws->next;
  register unsigned char c;
  register char const *end = p;
  register char const *lim = q;
  register int d = 1;
  register unsigned char const *y = kws->y;

  d = y[c = (end+=d)[-1]]; // no-warning
  trie = next[c];
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7593875> When handling sizeof(VLA) it leads to a hole in
// the ExplodedGraph (causing a false positive)
//===----------------------------------------------------------------------===//

int rdar_7593875_aux(int x);
int rdar_7593875(int n) {
  int z[n > 10 ? 10 : n]; // VLA.
  int v;
  v = rdar_7593875_aux(sizeof(z));
  // Previously we got a false positive about 'v' being uninitialized.
  return v; // no-warning
}

//===----------------------------------------------------------------------===//
// Handle casts from symbolic regions (packaged as integers) to doubles.
// Previously this caused an assertion failure.
//===----------------------------------------------------------------------===//

void *foo_rev95119(void);
void baz_rev95119(double x);
void bar_rev95119(void) {
  // foo_rev95119() returns a symbolic pointer.  It is then 
  // cast to an int which is then cast to a double.
  int value = (int) foo_rev95119();
  baz_rev95119((double)value);
}

//===----------------------------------------------------------------------===//
// Handle loading a symbolic pointer from a symbolic region that was
// invalidated by a call to an unknown function.
//===----------------------------------------------------------------------===//

void bar_rev95192(int **x);
void foo_rev95192(int **x) {
  *x = 0;
  bar_rev95192(x);
  // Not a null dereference.
  **x = 1; // no-warning
}

//===----------------------------------------------------------------------===//
// Handle casts of a function to a function pointer with a different return
// value.  We don't yet emit an error for such cases, but we now we at least
// don't crash when the return value gets interpreted in a way that
// violates our invariants.
//===----------------------------------------------------------------------===//

void *foo_rev95267(void);
int bar_rev95267(void) {
  char (*Callback_rev95267)(void) = (char (*)(void)) foo_rev95267;
  if ((*Callback_rev95267)() == (char) 0)
    return 1;
  return 0;
}

// Same as previous case, but handle casts to 'void'.
int bar_rev95274(void) {
  void (*Callback_rev95274)(void) = (void (*)(void)) foo_rev95267;
  (*Callback_rev95274)();
  return 0;
}

void rdar7582031_test_static_init_zero(void) {
  static unsigned x;
  if (x == 0)
    return;
  int *p = 0;
  *p = 0xDEADBEEF;
}
void rdar7582031_test_static_init_zero_b(void) {
  static void* x;
  if (x == 0)
    return;
  int *p = 0;
  *p = 0xDEADBEEF;
}

//===----------------------------------------------------------------------===//
// Test handling of parameters that are structs that contain floats and       //
// nested fields.                                                             //
//===----------------------------------------------------------------------===//

struct s_rev95547_nested { float x, y; };
struct s_rev95547 {
  struct s_rev95547_nested z1;
  struct s_rev95547_nested z2;
};
float foo_rev95547(struct s_rev95547 w) {
  return w.z1.x + 20.0; // no-warning
}
void foo_rev95547_b(struct s_rev95547 w) {
  struct s_rev95547 w2 = w;
  w2.z1.x += 20.0; // no-warning
}

//===----------------------------------------------------------------------===//
// Test handling statement expressions that don't populate a CFG block that
// is used to represent the computation of the RHS of a logical operator.
// This previously triggered a crash.
//===----------------------------------------------------------------------===//

void pr6938(void) {
  if (1 && ({
    while (0);
    0;
  }) == 0) {
  }
}

void pr6938_b(void) {
  if (1 && *({ // expected-warning{{Dereference of null pointer}}
    while (0) {}
    ({
      (int *) 0;
    });
  }) == 0) {
  }
}

//===----------------------------------------------------------------------===//
// <rdar://problem/7979430> - The CFG for code containing an empty
//  @synchronized block was previously broken (and would crash the analyzer).
//===----------------------------------------------------------------------===//

void r7979430(id x) {
  @synchronized(x) {}
}

//===----------------------------------------------------------------------===
// PR 7361 - Test that functions wrapped in macro instantiations are analyzed.
//===----------------------------------------------------------------------===
#define MAKE_TEST_FN() \
  void test_pr7361 (char a) {\
    char* b = 0x0;  *b = a;\
  }

MAKE_TEST_FN() // expected-warning{{null pointer}}

//===----------------------------------------------------------------------===
// PR 7491 - Test that symbolic expressions can be used as conditions.
//===----------------------------------------------------------------------===

void pr7491 (void) {
  extern int getint(void);
  int a = getint()-1;
  if (a) {
    return;
  }
  if (!a) {
    return;
  } else {
    // Should be unreachable
    (void)*(char*)0; // no-warning
  }
}

//===----------------------------------------------------------------------===
// PR 7475 - Test that assumptions about global variables are reset after
//  calling a global function.
//===----------------------------------------------------------------------===

int *pr7475_someGlobal;
void pr7475_setUpGlobal(void);

void pr7475(void) {
  if (pr7475_someGlobal == 0)
    pr7475_setUpGlobal();
  *pr7475_someGlobal = 0; // no-warning
}

void pr7475_warn(void) {
  static int *someStatic = 0;
  if (someStatic == 0)
    pr7475_setUpGlobal();
  *someStatic = 0; // expected-warning{{null pointer}}
}

// <rdar://problem/8202272> - __imag passed non-complex should not crash
float f0(_Complex float x) {
  float l0 = __real x;
  return  __real l0 + __imag l0;
}


//===----------------------------------------------------------------------===
// Test that we can reduce symbols to constants whether they are on the left
//  or right side of an expression.
//===----------------------------------------------------------------------===

void reduce_to_constant(int x, int y) {
  if (x != 20)
    return;

  int a = x + y;
  int b = y + x;

  if (y == -20 && a != 0)
    (void)*(char*)0; // no-warning
  if (y == -20 && b != 0)
    (void)*(char*)0; // no-warning
}

// <rdar://problem/8360854> - Test that code after a switch statement with no 
// 'case:' labels is correctly evaluated.
void r8360854(int n) {
  switch (n) {
   default: ;
  }
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{null pointer}}
}

// PR 8050 - crash in CastSizeChecker when pointee is an incomplete type
typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
void *malloc(size_t);

struct PR8050;

void pr8050(struct PR8050 **arg)
{
    *arg = malloc(1);
}

// <rdar://problem/5880430> Switch on enum should not consider default case live
//  if all enum values are covered
enum Cases { C1, C2, C3, C4 };
void test_enum_cases(enum Cases C) {
  switch (C) {
  case C1:
  case C2:
  case C4:
  case C3:
    return;
  }
  int *p = 0;
  *p = 0xDEADBEEF; // no-warning
}

void test_enum_cases_positive(enum Cases C) {
  switch (C) { // expected-warning{{enumeration value 'C4' not handled in switch}}
  case C1:
  case C2:
  case C3:
    return;
  }
  int *p = 0;
  *p = 0xDEADBEEF; // expected-warning{{Dereference of null pointer}}
}

// <rdar://problem/6351970> rule request: warn if synchronization mutex can be nil
void rdar6351970(void) {
  id x = 0;
  @synchronized(x) {} // expected-warning{{Nil value used as mutex for @synchronized() (no synchronization will occur)}}
}

void rdar6351970_b(id x) {
  if (!x)
    @synchronized(x) {} // expected-warning{{Nil value used as mutex for @synchronized() (no synchronization will occur)}}
}

void rdar6351970_c(void) {
  id x;
  @synchronized(x) {} // expected-warning{{Uninitialized value used as mutex for @synchronized}}
}

@interface Rdar8578650
- (id) foo8578650;
@end

void rdar8578650(id x) {
  @synchronized (x) {
    [x foo8578650];
  }
  // At this point we should assume that 'x' is not nil, not
  // the inverse.
  @synchronized (x) { // no-warning
  }
}

// <rdar://problem/6352035> rule request: direct structure member access null pointer dereference
@interface RDar6352035 {
  int c;
}
- (void)foo;
- (void)bar;
@end

@implementation RDar6352035
- (void)foo {
  RDar6352035 *friend = 0;
  friend->c = 7; // expected-warning{{Access to instance variable 'c' results in a dereference of a null pointer (loaded from variable 'friend')}}
}
- (void)bar {
  self = 0;
  c = 7; // expected-warning{{Access to instance variable 'c' results in a dereference of a null pointer (loaded from variable 'self')}}
}
@end

// PR 8149 - GNU statement expression in condition of ForStmt.
// This previously triggered an assertion failure in CFGBuilder.
void pr8149(void) {
  for (; ({ do { } while (0); 0; });) { }
}

// PR 8458 - Make sure @synchronized doesn't crash with properties.
@interface PR8458 {}
@property(readonly) id lock;
@end

static
void __PR8458(PR8458 *x) {
  @synchronized(x.lock) {} // no-warning
}

// PR 8440 - False null dereference during store to array-in-field-in-global.
// This test case previously resulted in a bogus null deref warning from
// incorrect lazy symbolication logic in RegionStore.
static struct {
  int num;
  char **data;
} saved_pr8440;

char *foo_pr8440(void);
char **bar_pr8440(void);
void baz_pr8440(int n)
{
   saved_pr8440.num = n;
   if (saved_pr8440.data) 
     return;
   saved_pr8440.data = bar_pr8440();
   for (int i = 0 ; i < n ; i ++)
     saved_pr8440.data[i] = foo_pr8440(); // no-warning
}

// Support direct accesses to non-null memory.  Reported in:
//  PR 5272
//  <rdar://problem/6839683>
int test_direct_address_load(void) {
  int *p = (int*) 0x4000;
  return *p; // no-warning
}

void pr5272_test(void) {
  struct pr5272 { int var2; };
  (*(struct pr5272*)0xBC000000).var2 = 0; // no-warning
  (*(struct pr5272*)0xBC000000).var2 += 2; // no-warning
}

// Support casting the return value of function to another different type
// This previously caused a crash, although we likely need more precise
// reasoning here. <rdar://problem/8663544>
void* rdar8663544(void);
typedef struct {} Val8663544;
Val8663544 bazR8663544(void) {
  Val8663544(*func) (void) = (Val8663544(*) (void)) rdar8663544;
  return func();
}

// PR 8619 - Handle ternary expressions with a call to a noreturn function.
// This previously resulted in a crash.
void pr8619_noreturn(int x) __attribute__((noreturn));

void pr8619(int a, int b, int c) {
  a ?: pr8619_noreturn(b || c);
}


// PR 8646 - crash in the analyzer when handling unions.
union pr8648_union {
        signed long long pr8648_union_field;
};
void pr8648(void) {
  long long y;
  union pr8648_union x = { .pr8648_union_field = 0LL };
  y = x.pr8648_union_field;
  
  union pr8648_union z;
  z = (union pr8648_union) { .pr8648_union_field = 0LL };

  union pr8648_union w;
  w = ({ (union pr8648_union) { .pr8648_union_field = 0LL }; }); 

  // crash, no assignment
  (void) ({ (union pr8648_union) { .pr8648_union_field = 0LL }; }).pr8648_union_field;

  // crash with assignment
  y = ({ (union pr8648_union) { .pr8648_union_field = 0LL }; }).pr8648_union_field;
}

// PR 9269 - don't assert when building the following CFG.  The for statement
// contains a condition with multiple basic blocks, and the value of the
// statement expression is then indexed as part of a bigger condition expression.
// This example exposed a bug in child traversal in the CFGBuilder.
void pr9269(void) {
  struct s { char *bar[10]; } baz[2] = { 0 };
  unsigned i = 0;
  for (i = 0;
  (* ({ while(0); ({ &baz[0]; }); })).bar[0] != 0; // expected-warning {{while loop has empty body}} expected-note {{put the semicolon on a separate line to silence this warning}}
       ++i) {}
}

// Test evaluation of GNU-style ?:.
int pr9287(int type) { return type ? : 0; } // no-warning

void pr9287_b(int type, int *p) { 
  int x = type ? : 0;
  if (x) {
    p = 0;
  }
  if (type) {
    *p = 0xDEADBEEF; // expected-warning {{null pointer}}
  }
}

void pr9287_c(int type, int *p) { 
  int x = type ? : 0;
  if (x) {
    p = 0;
  }
  if (!type) {
    *p = 0xDEADBEEF; // no-warning
  }
}

void test_switch(void) {
  switch (4) {
    case 1: {
      int *p = 0;
      *p = 0xDEADBEEF; // no-warning
      break;
    }
    case 4: {
      int *p = 0;
      *p = 0xDEADBEEF; // expected-warning {{null}}
      break;
    }
    default: {
      int *p = 0;
      *p = 0xDEADBEEF; // no-warning
      break;
    }
  }
}

// PR 9467.  Tests various CFG optimizations.  This previously crashed.
static void test(unsigned int bit_mask)
{
  unsigned int bit_index;
  for (bit_index = 0;
       bit_index < 24;
       bit_index++) {
    switch ((0x01 << bit_index) & bit_mask) {
    case 0x100000: ;
    }
  }
}

// Don't crash on code containing __label__.
int radar9414427_aux(void);
void radar9414427(void) {
  __label__ mylabel;
  if (radar9414427_aux()) {
  mylabel: do {}
  while (0);
  }
}

// Analyze methods in @implementation (category)
@interface RDar9465344
@end

@implementation RDar9465344 (MyCategory)
- (void) testcategoryImpl {
  int *p = 0x0;
  *p = 0xDEADBEEF; // expected-warning {{null}}
}
@end

@implementation RDar9465344
@end

// Don't crash when analyzing access to 'self' within a block.
@interface Rdar10380300Base 
- (void) foo;
@end
@interface Rdar10380300 : Rdar10380300Base @end
@implementation Rdar10380300
- (void)foo {
  ^{
    [super foo];
  }();
}
@end

// Don't crash when a ?: is only preceded by a statement (not an expression)
// in the CFG.
void __assert_fail(void);

enum rdar1196620_e { E_A, E_B, E_C, E_D };
struct rdar1196620_s { int ints[E_D+1]; };

static void rdar1196620_call_assert(struct rdar1196620_s* s) {
  int i = 0;
  s?(void)0:__assert_fail();
}

static void rdar1196620(struct rdar1196620_s* s) {
  rdar1196620_call_assert(s);
}


