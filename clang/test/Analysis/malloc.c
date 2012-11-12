// RUN: %clang_cc1 -analyze -analyzer-checker=core,alpha.deadcode.UnreachableCode,alpha.core.CastSize,unix.Malloc,debug.ExprInspection -analyzer-store=region -verify %s
#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *valloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *reallocf(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
char *strdup(const char *s);
char *strndup(const char *s, size_t n);

void myfoo(int *p);
void myfooint(int p);
char *fooRetPtr();

void f1() {
  int *p = malloc(12);
  return; // expected-warning{{Memory is never released; potential leak of memory pointed to by 'p'}}
}

void f2() {
  int *p = malloc(12);
  free(p);
  free(p); // expected-warning{{Attempt to free released memory}}
}

void f2_realloc_0() {
  int *p = malloc(12);
  realloc(p,0);
  realloc(p,0); // expected-warning{{Attempt to free released memory}}
}

void f2_realloc_1() {
  int *p = malloc(12);
  int *q = realloc(p,0); // no-warning
}

void reallocNotNullPtr(unsigned sizeIn) {
  unsigned size = 12;
  char *p = (char*)malloc(size);
  if (p) {
    char *q = (char*)realloc(p, sizeIn);
    char x = *q; // expected-warning {{Memory is never released; potential leak of memory pointed to by 'q'}}
  }
}

int *realloctest1() {
  int *q = malloc(12);
  q = realloc(q, 20);
  return q; // no warning - returning the allocated value
}

// p should be freed if realloc fails.
void reallocFails() {
  char *p = malloc(12);
  char *r = realloc(p, 12+1);
  if (!r) {
    free(p);
  } else {
    free(r);
  }
}

void reallocSizeZero1() {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  if (!r) {
    free(p); // expected-warning {{Attempt to free released memory}}
  } else {
    free(r);
  }
}

void reallocSizeZero2() {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  if (!r) {
    free(p); // expected-warning {{Attempt to free released memory}}
  } else {
    free(r);
  }
  free(p); // expected-warning {{Attempt to free released memory}}
}

void reallocSizeZero3() {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  free(r);
}

void reallocSizeZero4() {
  char *r = realloc(0, 0);
  free(r);
}

void reallocSizeZero5() {
  char *r = realloc(0, 0);
}

void reallocPtrZero1() {
  char *r = realloc(0, 12); // expected-warning {{Memory is never released; potential leak of memory pointed to by 'r'}}
}

void reallocPtrZero2() {
  char *r = realloc(0, 12);
  if (r)
    free(r);
}

void reallocPtrZero3() {
  char *r = realloc(0, 12);
  free(r);
}

void reallocRadar6337483_1() {
    char *buf = malloc(100);
    buf = (char*)realloc(buf, 0x1000000);
    if (!buf) {
        return;// expected-warning {{Memory is never released; potential leak}}
    }
    free(buf);
}

void reallocRadar6337483_2() {
    char *buf = malloc(100);
    char *buf2 = (char*)realloc(buf, 0x1000000);
    if (!buf2) { // expected-warning {{Memory is never released; potential leak}}
      ;
    } else {
      free(buf2);
    }
}

void reallocRadar6337483_3() {
    char * buf = malloc(100);
    char * tmp;
    tmp = (char*)realloc(buf, 0x1000000);
    if (!tmp) {
        free(buf);
        return;
    }
    buf = tmp;
    free(buf);
}

void reallocRadar6337483_4() {
    char *buf = malloc(100);
    char *buf2 = (char*)realloc(buf, 0x1000000);
    if (!buf2) {
      return;  // expected-warning {{Memory is never released; potential leak}}
    } else {
      free(buf2);
    }
}

int *reallocfTest1() {
  int *q = malloc(12);
  q = reallocf(q, 20);
  return q; // no warning - returning the allocated value
}

void reallocfRadar6337483_4() {
    char *buf = malloc(100);
    char *buf2 = (char*)reallocf(buf, 0x1000000);
    if (!buf2) {
      return;  // no warning - reallocf frees even on failure
    } else {
      free(buf2);
    }
}

void reallocfRadar6337483_3() {
    char * buf = malloc(100);
    char * tmp;
    tmp = (char*)reallocf(buf, 0x1000000);
    if (!tmp) {
        free(buf); // expected-warning {{Attempt to free released memory}}
        return;
    }
    buf = tmp;
    free(buf);
}

void reallocfPtrZero1() {
  char *r = reallocf(0, 12); // expected-warning {{Memory is never released; potential leak}}
}


// This case tests that storing malloc'ed memory to a static variable which is
// then returned is not leaked.  In the absence of known contracts for functions
// or inter-procedural analysis, this is a conservative answer.
int *f3() {
  static int *p = 0;
  p = malloc(12); 
  return p; // no-warning
}

// This case tests that storing malloc'ed memory to a static global variable
// which is then returned is not leaked.  In the absence of known contracts for
// functions or inter-procedural analysis, this is a conservative answer.
static int *p_f4 = 0;
int *f4() {
  p_f4 = malloc(12); 
  return p_f4; // no-warning
}

int *f5() {
  int *q = malloc(12);
  q = realloc(q, 20);
  return q; // no-warning
}

void f6() {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    free(p);
}

void f6_realloc() {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    realloc(p,0);
}


char *doit2();
void pr6069() {
  char *buf = doit2();
  free(buf);
}

void pr6293() {
  free(0);
}

void f7() {
  char *x = (char*) malloc(4);
  free(x);
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
}

void f8() {
  char *x = (char*) malloc(4);
  free(x);
  char *y = strndup(x, 4); // expected-warning{{Use of memory after it is freed}}
}

void f7_realloc() {
  char *x = (char*) malloc(4);
  realloc(x,0);
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
}

void PR6123() {
  int *x = malloc(11); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
}

void PR7217() {
  int *buf = malloc(2); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  buf[1] = 'c'; // not crash
}

void mallocCastToVoid() {
  void *p = malloc(2);
  const void *cp = p; // not crash
  free(p);
}

void mallocCastToFP() {
  void *p = malloc(2);
  void (*fp)() = p; // not crash
  free(p);
}

// This tests that malloc() buffers are undefined by default
char mallocGarbage () {
	char *buf = malloc(2);
	char result = buf[1]; // expected-warning{{undefined}}
	free(buf);
	return result;
}

// This tests that calloc() buffers need to be freed
void callocNoFree () {
  char *buf = calloc(2,2);
  return; // expected-warning{{never released}}
}

// These test that calloc() buffers are zeroed by default
char callocZeroesGood () {
	char *buf = calloc(2,2);
	char result = buf[3]; // no-warning
	if (buf[1] == 0) {
	  free(buf);
	}
	return result; // no-warning
}

char callocZeroesBad () {
	char *buf = calloc(2,2);
	char result = buf[3]; // no-warning
	if (buf[1] != 0) {
	  free(buf); // expected-warning{{never executed}}
	}
	return result; // expected-warning{{never released}}
}

void nullFree() {
  int *p = 0;
  free(p); // no warning - a nop
}

void paramFree(int *p) {
  myfoo(p);
  free(p); // no warning
  myfoo(p); // expected-warning {{Use of memory after it is freed}}
}

int* mallocEscapeRet() {
  int *p = malloc(12);
  return p; // no warning
}

void mallocEscapeFoo() {
  int *p = malloc(12);
  myfoo(p);
  return; // no warning
}

void mallocEscapeFree() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
}

void mallocEscapeFreeFree() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  free(p); // expected-warning{{Attempt to free released memory}}
}

void mallocEscapeFreeUse() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  myfoo(p); // expected-warning{{Use of memory after it is freed}}
}

int *myalloc();
void myalloc2(int **p);

void mallocEscapeFreeCustomAlloc() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  p = myalloc();
  free(p); // no warning
}

void mallocEscapeFreeCustomAlloc2() {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  myalloc2(&p);
  free(p); // no warning
}

void mallocBindFreeUse() {
  int *x = malloc(12);
  int *y = x;
  free(y);
  myfoo(x); // expected-warning{{Use of memory after it is freed}}
}

void mallocEscapeMalloc() {
  int *p = malloc(12);
  myfoo(p);
  p = malloc(12); // expected-warning{{Memory is never released; potential leak}}
}

void mallocMalloc() {
  int *p = malloc(12);
  p = malloc(12); // expected-warning {{Memory is never released; potential leak}}
}

void mallocFreeMalloc() {
  int *p = malloc(12);
  free(p);
  p = malloc(12);
  free(p);
}

void mallocFreeUse_params() {
  int *p = malloc(12);
  free(p);
  myfoo(p); //expected-warning{{Use of memory after it is freed}}
}

void mallocFreeUse_params2() {
  int *p = malloc(12);
  free(p);
  myfooint(*p); //expected-warning{{Use of memory after it is freed}}
}

void mallocFailedOrNot() {
  int *p = malloc(12);
  if (!p)
    free(p);
  else
    free(p);
}

struct StructWithInt {
  int g;
};

int *mallocReturnFreed() {
  int *p = malloc(12);
  free(p);
  return p; // expected-warning {{Use of memory after it is freed}}
}

int useAfterFreeStruct() {
  struct StructWithInt *px= malloc(sizeof(struct StructWithInt));
  px->g = 5;
  free(px);
  return px->g; // expected-warning {{Use of memory after it is freed}}
}

void nonSymbolAsFirstArg(int *pp, struct StructWithInt *p);

void mallocEscapeFooNonSymbolArg() {
  struct StructWithInt *p = malloc(sizeof(struct StructWithInt));
  nonSymbolAsFirstArg(&p->g, p);
  return; // no warning
}

void mallocFailedOrNotLeak() {
  int *p = malloc(12);
  if (p == 0)
    return; // no warning
  else
    return; // expected-warning {{Memory is never released; potential leak}}
}

void mallocAssignment() {
  char *p = malloc(12);
  p = fooRetPtr(); // expected-warning {{leak}}
}

int vallocTest() {
  char *mem = valloc(12);
  return 0; // expected-warning {{Memory is never released; potential leak}}
}

void vallocEscapeFreeUse() {
  int *p = valloc(12);
  myfoo(p);
  free(p);
  myfoo(p); // expected-warning{{Use of memory after it is freed}}
}

int *Gl;
struct GlStTy {
  int *x;
};

struct GlStTy GlS = {0};

void GlobalFree() {
  free(Gl);
}

void GlobalMalloc() {
  Gl = malloc(12);
}

void GlobalStructMalloc() {
  int *a = malloc(12);
  GlS.x = a;
}

void GlobalStructMallocFree() {
  int *a = malloc(12);
  GlS.x = a;
  free(GlS.x);
}

char *ArrayG[12];

void globalArrayTest() {
  char *p = (char*)malloc(12);
  ArrayG[0] = p;
}

// Make sure that we properly handle a pointer stored into a local struct/array.
typedef struct _StructWithPtr {
  int *memP;
} StructWithPtr;

static StructWithPtr arrOfStructs[10];

void testMalloc() {
  int *x = malloc(12);
  StructWithPtr St;
  St.memP = x;
  arrOfStructs[0] = St; // no-warning
}

StructWithPtr testMalloc2() {
  int *x = malloc(12);
  StructWithPtr St;
  St.memP = x;
  return St; // no-warning
}

int *testMalloc3() {
  int *x = malloc(12);
  int *y = x;
  return y; // no-warning
}

void testStructLeak() {
  StructWithPtr St;
  St.memP = malloc(12);
  return; // expected-warning {{Memory is never released; potential leak of memory pointed to by 'St.memP'}}
}

void testElemRegion1() {
  char *x = (void*)malloc(2);
  int *ix = (int*)x;
  free(&(x[0]));
}

void testElemRegion2(int **pp) {
  int *p = malloc(12);
  *pp = p;
  free(pp[0]);
}

void testElemRegion3(int **pp) {
  int *p = malloc(12);
  *pp = p;
  free(*pp);
}
// Region escape testing.

unsigned takePtrToPtr(int **p);
void PassTheAddrOfAllocatedData(int f) {
  int *p = malloc(12);
  // We don't know what happens after the call. Should stop tracking here.
  if (takePtrToPtr(&p))
    f++;
  free(p); // no warning
}

struct X {
  int *p;
};
unsigned takePtrToStruct(struct X *s);
int ** foo2(int *g, int f) {
  int *p = malloc(12);
  struct X *px= malloc(sizeof(struct X));
  px->p = p;
  // We don't know what happens after this call. Should not track px nor p.
  if (takePtrToStruct(px))
    f++;
  free(p);
  return 0;
}

struct X* RegInvalidationDetect1(struct X *s2) {
  struct X *px= malloc(sizeof(struct X));
  px->p = 0;
  px = s2;
  return px; // expected-warning {{Memory is never released; potential leak}}
}

struct X* RegInvalidationGiveUp1() {
  int *p = malloc(12);
  struct X *px= malloc(sizeof(struct X));
  px->p = p;
  return px;
}

int **RegInvalidationDetect2(int **pp) {
  int *p = malloc(12);
  pp = &p;
  pp++;
  return 0;// expected-warning {{Memory is never released; potential leak}}
}

extern void exit(int) __attribute__ ((__noreturn__));
void mallocExit(int *g) {
  struct xx *p = malloc(12);
  if (g != 0)
    exit(1);
  free(p);
  return;
}

extern void __assert_fail (__const char *__assertion, __const char *__file,
    unsigned int __line, __const char *__function)
     __attribute__ ((__noreturn__));
#define assert(expr) \
  ((expr)  ? (void)(0)  : __assert_fail (#expr, __FILE__, __LINE__, __func__))
void mallocAssert(int *g) {
  struct xx *p = malloc(12);

  assert(g != 0);
  free(p);
  return;
}

void doNotInvalidateWhenPassedToSystemCalls(char *s) {
  char *p = malloc(12);
  strlen(p);
  strcpy(p, s); // expected-warning {{leak}}
}

// Rely on the CString checker evaluation of the strcpy API to convey that the result of strcpy is equal to p.
void symbolLostWithStrcpy(char *s) {
  char *p = malloc(12);
  p = strcpy(p, s);
  free(p);
}


// The same test as the one above, but with what is actually generated on a mac.
static __inline char *
__inline_strcpy_chk (char *restrict __dest, const char *restrict __src)
{
  return __builtin___strcpy_chk (__dest, __src, __builtin_object_size (__dest, 2 > 1));
}

void symbolLostWithStrcpy_InlineStrcpyVersion(char *s) {
  char *p = malloc(12);
  p = ((__builtin_object_size (p, 0) != (size_t) -1) ? __builtin___strcpy_chk (p, s, __builtin_object_size (p, 2 > 1)) : __inline_strcpy_chk (p, s));
  free(p);
}

// Here we are returning a pointer one past the allocated value. An idiom which
// can be used for implementing special malloc. The correct uses of this might
// be rare enough so that we could keep this as a warning.
static void *specialMalloc(int n){
  int *p;
  p = malloc( n+8 );
  if( p ){
    p[0] = n;
    p++;
  }
  return p;
}

// Potentially, the user could free the struct by performing pointer arithmetic on the return value.
// This is a variation of the specialMalloc issue, though probably would be more rare in correct code.
int *specialMallocWithStruct() {
  struct StructWithInt *px= malloc(sizeof(struct StructWithInt));
  return &(px->g);
}

// Test various allocation/deallocation functions.
void testStrdup(const char *s, unsigned validIndex) {
  char *s2 = strdup(s);
  s2[validIndex + 1] = 'b';// expected-warning {{Memory is never released; potential leak}}
}

int testStrndup(const char *s, unsigned validIndex, unsigned size) {
  char *s2 = strndup(s, size);
  s2 [validIndex + 1] = 'b';
  if (s2[validIndex] != 'a')
    return 0;
  else
    return 1;// expected-warning {{Memory is never released; potential leak}}
}

void testStrdupContentIsDefined(const char *s, unsigned validIndex) {
  char *s2 = strdup(s);
  char result = s2[1];// no warning
  free(s2);
}

// ----------------------------------------------------------------------------
// Test the system library functions to which the pointer can escape.
// This tests false positive suppression.

// For now, we assume memory passed to pthread_specific escapes.
// TODO: We could check that if a new pthread binding is set, the existing
// binding must be freed; otherwise, a memory leak can occur.
void testPthereadSpecificEscape(pthread_key_t key) {
  void *buf = malloc(12);
  pthread_setspecific(key, buf); // no warning
}

// PR12101: Test funopen().
static int releasePtr(void *_ctx) {
    free(_ctx);
    return 0;
}
FILE *useFunOpen() {
    void *ctx = malloc(sizeof(int));
    FILE *f = funopen(ctx, 0, 0, 0, releasePtr); // no warning
    if (f == 0) {
        free(ctx);
    }
    return f;
}
FILE *useFunOpenNoReleaseFunction() {
    void *ctx = malloc(sizeof(int));
    FILE *f = funopen(ctx, 0, 0, 0, 0);
    if (f == 0) {
        free(ctx);
    }
    return f; // expected-warning{{leak}}
}

static int readNothing(void *_ctx, char *buf, int size) {
  return 0;
}
FILE *useFunOpenReadNoRelease() {
  void *ctx = malloc(sizeof(int));
  FILE *f = funopen(ctx, readNothing, 0, 0, 0);
  if (f == 0) {
    free(ctx);
  }
  return f; // expected-warning{{leak}}
}

// Test setbuf, setvbuf.
int my_main_no_warning() {
    char *p = malloc(100);
    setvbuf(stdout, p, 0, 100);
    return 0;
}
int my_main_no_warning2() {
    char *p = malloc(100);
    setbuf(__stdoutp, p);
    return 0;
}
int my_main_warn(FILE *f) {
    char *p = malloc(100);
    setvbuf(f, p, 0, 100);
    return 0;// expected-warning {{leak}}
}

// <rdar://problem/10978247>.
// some people use stack allocated memory as an optimization to avoid
// a heap allocation for small work sizes.  This tests the analyzer's
// understanding that the malloc'ed memory is not the same as stackBuffer.
void radar10978247(int myValueSize) {
  char stackBuffer[128];
  char *buffer;

  if (myValueSize <= sizeof(stackBuffer))
    buffer = stackBuffer;
  else 
    buffer = malloc(myValueSize);

  // do stuff with the buffer
  if (buffer != stackBuffer)
    free(buffer);
}

void radar10978247_positive(int myValueSize) {
  char stackBuffer[128];
  char *buffer;

  if (myValueSize <= sizeof(stackBuffer))
    buffer = stackBuffer;
  else 
    buffer = malloc(myValueSize);

  // do stuff with the buffer
  if (buffer == stackBuffer) // expected-warning {{leak}}
    return;
}

// <rdar://problem/11269741> Previously this triggered a false positive
// because malloc() is known to return uninitialized memory and the binding
// of 'o' to 'p->n' was not getting propertly handled.  Now we report a leak.
struct rdar11269741_a_t {
  struct rdar11269741_b_t {
    int m;
  } n;
};

int rdar11269741(struct rdar11269741_b_t o)
{
  struct rdar11269741_a_t *p = (struct rdar11269741_a_t *) malloc(sizeof(*p));
  p->n = o;
  return p->n.m; // expected-warning {{leak}}
}

// Pointer arithmetic, returning an ElementRegion.
void *radar11329382(unsigned bl) {
  void *ptr = malloc (16);
  ptr = ptr + (2 - bl);
  return ptr; // no warning
}

void __assert_rtn(const char *, const char *, int, const char *) __attribute__((__noreturn__));
int strcmp(const char *, const char *);
char *a (void);
void radar11270219(void) {
  char *x = a(), *y = a();
  (__builtin_expect(!(x && y), 0) ? __assert_rtn(__func__, "/Users/zaks/tmp/ex.c", 24, "x && y") : (void)0);
  strcmp(x, y); // no warning
}

void radar_11358224_test_double_assign_ints_positive_2()
{
  void *ptr = malloc(16);
  ptr = ptr; // expected-warning {{leak}}
}

// Assume that functions which take a function pointer can free memory even if
// they are defined in system headers and take the const pointer to the
// allocated memory. (radar://11160612)
int const_ptr_and_callback(int, const char*, int n, void(*)(void*));
void r11160612_1() {
  char *x = malloc(12);
  const_ptr_and_callback(0, x, 12, free); // no - warning
}

// Null is passed as callback.
void r11160612_2() {
  char *x = malloc(12);
  const_ptr_and_callback(0, x, 12, 0); // expected-warning {{leak}}
}

// Callback is passed to a function defined in a system header.
void r11160612_4() {
  char *x = malloc(12);
  sqlite3_bind_text_my(0, x, 12, free); // no - warning
}

// Passing callbacks in a struct.
void r11160612_5(StWithCallback St) {
  void *x = malloc(12);
  dealocateMemWhenDoneByVal(x, St);
}
void r11160612_6(StWithCallback St) {
  void *x = malloc(12);
  dealocateMemWhenDoneByRef(&St, x);
}

int mySub(int, int);
int myAdd(int, int);
int fPtr(unsigned cond, int x) {
  return (cond ? mySub : myAdd)(x, x);
}

// Test anti-aliasing.

void dependsOnValueOfPtr(int *g, unsigned f) {
  int *p;

  if (f) {
    p = g;
  } else {
    p = malloc(12);
  }

  if (p != g)
    free(p);
  else
    return; // no warning
  return;
}

int CMPRegionHeapToStack() {
  int x = 0;
  int *x1 = malloc(8);
  int *x2 = &x;
  clang_analyzer_eval(x1 == x2); // expected-warning{{FALSE}}
  free(x1);
  return x;
}

int CMPRegionHeapToHeap2() {
  int x = 0;
  int *x1 = malloc(8);
  int *x2 = malloc(8);
  int *x4 = x1;
  int *x5 = x2;
  clang_analyzer_eval(x4 == x5); // expected-warning{{FALSE}}
  free(x1);
  free(x2);
  return x;
}

int CMPRegionHeapToHeap() {
  int x = 0;
  int *x1 = malloc(8);
  int *x4 = x1;
  if (x1 == x4) {
    free(x1);
    return 5/x; // expected-warning{{Division by zero}}
  }
  return x;// expected-warning{{This statement is never executed}}
}

int HeapAssignment() {
  int m = 0;
  int *x = malloc(4);
  int *y = x;
  *x = 5;
  clang_analyzer_eval(*x != *y); // expected-warning{{FALSE}}
  free(x);
  return 0;
}

int *retPtr();
int *retPtrMightAlias(int *x);
int cmpHeapAllocationToUnknown() {
  int zero = 0;
  int *yBefore = retPtr();
  int *m = malloc(8);
  int *yAfter = retPtrMightAlias(m);
  clang_analyzer_eval(yBefore == m); // expected-warning{{FALSE}}
  clang_analyzer_eval(yAfter == m); // expected-warning{{FALSE}}
  free(m);
  return 0;
}

void localArrayTest() {
  char *p = (char*)malloc(12);
  char *ArrayL[12];
  ArrayL[0] = p; // expected-warning {{leak}}
}

void localStructTest() {
  StructWithPtr St;
  StructWithPtr *pSt = &St;
  pSt->memP = malloc(12); // expected-warning{{Memory is never released; potential leak}}
}

// Test double assignment through integers.
static long glob;
void test_double_assign_ints()
{
  void *ptr = malloc (16);  // no-warning
  glob = (long)(unsigned long)ptr;
}

void test_double_assign_ints_positive()
{
  void *ptr = malloc(16);
  (void*)(long)(unsigned long)ptr; // expected-warning {{unused}} expected-warning {{leak}}
}


void testCGContextNoLeak()
{
  void *ptr = malloc(16);
  CGContextRef context = CGBitmapContextCreate(ptr);

  // Because you can get the data back out like this, even much later,
  // CGBitmapContextCreate is one of our "stop-tracking" exceptions.
  free(CGBitmapContextGetData(context));
}

void testCGContextLeak()
{
  void *ptr = malloc(16);
  CGContextRef context = CGBitmapContextCreate(ptr);
  // However, this time we're just leaking the data, because the context
  // object doesn't escape and it hasn't been freed in this function.
}

// Allow xpc context to escape. radar://11635258
// TODO: Would be great if we checked that the finalize_connection_context actually releases it.
static void finalize_connection_context(void *ctx) {
  int *context = ctx;
  free(context);
}
void foo (xpc_connection_t peer) {
  int *ctx = calloc(1, sizeof(int));
  xpc_connection_set_context(peer, ctx);
  xpc_connection_set_finalizer_f(peer, finalize_connection_context);
  xpc_connection_resume(peer);
}

// Make sure we catch errors when we free in a function which does not allocate memory.
void freeButNoMalloc(int *p, int x){
  if (x) {
    free(p);
    //user forgot a return here.
  }
  free(p); // expected-warning {{Attempt to free released memory}}
}

struct HasPtr {
  char *p;
};

char* reallocButNoMalloc(struct HasPtr *a, int c, int size) {
  int *s;
  char *b = realloc(a->p, size);
  char *m = realloc(a->p, size); // expected-warning {{Attempt to free released memory}}
  return a->p;
}

// We should not warn in this case since the caller will presumably free a->p in all cases.
int reallocButNoMallocPR13674(struct HasPtr *a, int c, int size) {
  int *s;
  char *b = realloc(a->p, size);
  if (b == 0)
    return -1;
  a->p = b;
  return 0;
}

// Test realloc with no visible malloc.
void *test(void *ptr) {
  void *newPtr = realloc(ptr, 4);
  if (newPtr == 0) {
    if (ptr)
      free(ptr); // no-warning
  }
  return newPtr;
}

// ----------------------------------------------------------------------------
// False negatives.

// TODO: This is another false negative.
void testMallocWithParam(int **p) {
  *p = (int*) malloc(sizeof(int));
  *p = 0;
}

void testMallocWithParam_2(int **p) {
  *p = (int*) malloc(sizeof(int));
}
