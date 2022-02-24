// RUN: %clang_analyze_cc1 -Wno-strict-prototypes -analyzer-store=region -verify %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.deadcode.UnreachableCode \
// RUN:   -analyzer-checker=alpha.core.CastSize \
// RUN:   -analyzer-checker=unix \
// RUN:   -analyzer-checker=debug.ExprInspection

#include "Inputs/system-header-simulator.h"

void clang_analyzer_eval(int);
void clang_analyzer_dump(int);
void clang_analyzer_dumpExtent(void *);

// Without -fms-compatibility, wchar_t isn't a builtin type. MSVC defines
// _WCHAR_T_DEFINED if wchar_t is available. Microsoft recommends that you use
// the builtin type: "Using the typedef version can cause portability
// problems", but we're ok here because we're not actually running anything.
// Also of note is this cryptic warning: "The wchar_t type is not supported
// when you compile C code".
//
// See the docs for more:
// https://msdn.microsoft.com/en-us/library/dh8che7s.aspx
#if !defined(_WCHAR_T_DEFINED)
// "Microsoft implements wchar_t as a two-byte unsigned value"
typedef unsigned short wchar_t;
#define _WCHAR_T_DEFINED
#endif // !defined(_WCHAR_T_DEFINED)

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *alloca(size_t);
void *valloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *reallocf(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
char *strdup(const char *s);
wchar_t *wcsdup(const wchar_t *s);
char *strndup(const char *s, size_t n);
int memcmp(const void *s1, const void *s2, size_t n);

// Windows variants
char *_strdup(const char *strSource);
wchar_t *_wcsdup(const wchar_t *strSource);
void *_alloca(size_t size);

void myfoo(int *p);
void myfooint(int p);
char *fooRetPtr(void);

void f1(void) {
  int *p = malloc(12);
  return; // expected-warning{{Potential leak of memory pointed to by 'p'}}
}

void f2(void) {
  int *p = malloc(12);
  free(p);
  free(p); // expected-warning{{Attempt to free released memory}}
}

void f2_realloc_0(void) {
  int *p = malloc(12);
  realloc(p,0);
  realloc(p,0); // expected-warning{{Attempt to free released memory}}
}

void f2_realloc_1(void) {
  int *p = malloc(12);
  int *q = realloc(p,0); // no-warning
}

void reallocNotNullPtr(unsigned sizeIn) {
  unsigned size = 12;
  char *p = (char*)malloc(size);
  if (p) {
    char *q = (char*)realloc(p, sizeIn);
    char x = *q; // expected-warning {{Potential leak of memory pointed to by 'q'}}
  }
}

void allocaTest(void) {
  int *p = alloca(sizeof(int));
} // no warn

void winAllocaTest(void) {
  int *p = _alloca(sizeof(int));
} // no warn

void allocaBuiltinTest(void) {
  int *p = __builtin_alloca(sizeof(int));
} // no warn

int *realloctest1(void) {
  int *q = malloc(12);
  q = realloc(q, 20);
  return q; // no warning - returning the allocated value
}

// p should be freed if realloc fails.
void reallocFails(void) {
  char *p = malloc(12);
  char *r = realloc(p, 12+1);
  if (!r) {
    free(p);
  } else {
    free(r);
  }
}

void reallocSizeZero1(void) {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  if (!r) {
    free(p); // expected-warning {{Attempt to free released memory}}
  } else {
    free(r);
  }
}

void reallocSizeZero2(void) {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  if (!r) {
    free(p); // expected-warning {{Attempt to free released memory}}
  } else {
    free(r);
  }
  free(p); // expected-warning {{Attempt to free released memory}}
}

void reallocSizeZero3(void) {
  char *p = malloc(12);
  char *r = realloc(p, 0);
  free(r);
}

void reallocSizeZero4(void) {
  char *r = realloc(0, 0);
  free(r);
}

void reallocSizeZero5(void) {
  char *r = realloc(0, 0);
}

void reallocPtrZero1(void) {
  char *r = realloc(0, 12);
} // expected-warning {{Potential leak of memory pointed to by 'r'}}

void reallocPtrZero2(void) {
  char *r = realloc(0, 12);
  if (r)
    free(r);
}

void reallocPtrZero3(void) {
  char *r = realloc(0, 12);
  free(r);
}

void reallocRadar6337483_1(void) {
    char *buf = malloc(100);
    buf = (char*)realloc(buf, 0x1000000);
    if (!buf) {
        return;// expected-warning {{Potential leak of memory pointed to by}}
    }
    free(buf);
}

void reallocRadar6337483_2(void) {
    char *buf = malloc(100);
    char *buf2 = (char*)realloc(buf, 0x1000000);
    if (!buf2) {
      ;
    } else {
      free(buf2);
    }
} // expected-warning {{Potential leak of memory pointed to by}}

void reallocRadar6337483_3(void) {
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

void reallocRadar6337483_4(void) {
    char *buf = malloc(100);
    char *buf2 = (char*)realloc(buf, 0x1000000);
    if (!buf2) {
      return;  // expected-warning {{Potential leak of memory pointed to by}}
    } else {
      free(buf2);
    }
}

int *reallocfTest1(void) {
  int *q = malloc(12);
  q = reallocf(q, 20);
  return q; // no warning - returning the allocated value
}

void reallocfRadar6337483_4(void) {
    char *buf = malloc(100);
    char *buf2 = (char*)reallocf(buf, 0x1000000);
    if (!buf2) {
      return;  // no warning - reallocf frees even on failure
    } else {
      free(buf2);
    }
}

void reallocfRadar6337483_3(void) {
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

void reallocfPtrZero1(void) {
  char *r = reallocf(0, 12);
} // expected-warning {{Potential leak of memory pointed to by}}

//------------------- Check usage of zero-allocated memory ---------------------
void CheckUseZeroAllocatedNoWarn1(void) {
  int *p = malloc(0);
  free(p); // no warning
}

void CheckUseZeroAllocatedNoWarn2(void) {
  int *p = alloca(0); // no warning
}

void CheckUseZeroWinAllocatedNoWarn2(void) {
  int *p = _alloca(0); // no warning
}


void CheckUseZeroAllocatedNoWarn3(void) {
  int *p = malloc(0);
  int *q = realloc(p, 8); // no warning
  free(q);
}

void CheckUseZeroAllocatedNoWarn4(void) {
  int *p = realloc(0, 8);
  *p = 1; // no warning
  free(p);
}

void CheckUseZeroAllocated1(void) {
  int *p = malloc(0);
  *p = 1; // expected-warning {{Use of memory allocated with size zero}}
  free(p);
}

char CheckUseZeroAllocated2(void) {
  char *p = alloca(0);
  return *p; // expected-warning {{Use of memory allocated with size zero}}
}

char CheckUseZeroWinAllocated2(void) {
  char *p = _alloca(0);
  return *p; // expected-warning {{Use of memory allocated with size zero}}
}

void UseZeroAllocated(int *p) {
  if (p)
    *p = 7; // expected-warning {{Use of memory allocated with size zero}}
}
void CheckUseZeroAllocated3(void) {
  int *p = malloc(0);
  UseZeroAllocated(p);
}

void f(char);
void CheckUseZeroAllocated4(void) {
  char *p = valloc(0);
  f(*p); // expected-warning {{Use of memory allocated with size zero}}
  free(p);
}

void CheckUseZeroAllocated5(void) {
  int *p = calloc(0, 2);
  *p = 1; // expected-warning {{Use of memory allocated with size zero}}
  free(p);
}

void CheckUseZeroAllocated6(void) {
  int *p = calloc(2, 0);
  *p = 1; // expected-warning {{Use of memory allocated with size zero}}
  free(p);
}

void CheckUseZeroAllocated7(void) {
  int *p = realloc(0, 0);
  *p = 1; // expected-warning {{Use of memory allocated with size zero}}
  free(p);
}

void CheckUseZeroAllocated8(void) {
  int *p = malloc(8);
  int *q = realloc(p, 0);
  *q = 1; // expected-warning {{Use of memory allocated with size zero}}
  free(q);
}

void CheckUseZeroAllocated9(void) {
  int *p = realloc(0, 0);
  int *q = realloc(p, 0);
  *q = 1; // expected-warning {{Use of memory allocated with size zero}}
  free(q);
}

void CheckUseZeroAllocatedPathNoWarn(_Bool b) {
  int s = 0;
  if (b)
    s= 10;

  char *p = malloc(s);

  if (b)
    *p = 1; // no warning

  free(p);
}

void CheckUseZeroAllocatedPathWarn(_Bool b) {
  int s = 10;
  if (b)
    s= 0;

  char *p = malloc(s);

  if (b)
    *p = 1; // expected-warning {{Use of memory allocated with size zero}}

  free(p);
}

void CheckUseZeroReallocatedPathNoWarn(_Bool b) {
  int s = 0;
  if (b)
    s= 10;

  char *p = malloc(8);
  char *q = realloc(p, s);

  if (b)
    *q = 1; // no warning

  free(q);
}

void CheckUseZeroReallocatedPathWarn(_Bool b) {
  int s = 10;
  if (b)
    s= 0;

  char *p = malloc(8);
  char *q = realloc(p, s);

  if (b)
    *q = 1; // expected-warning {{Use of memory allocated with size zero}}

  free(q);
}

// This case tests that storing malloc'ed memory to a static variable which is
// then returned is not leaked.  In the absence of known contracts for functions
// or inter-procedural analysis, this is a conservative answer.
int *f3(void) {
  static int *p = 0;
  p = malloc(12);
  return p; // no-warning
}

// This case tests that storing malloc'ed memory to a static global variable
// which is then returned is not leaked.  In the absence of known contracts for
// functions or inter-procedural analysis, this is a conservative answer.
static int *p_f4 = 0;
int *f4(void) {
  p_f4 = malloc(12);
  return p_f4; // no-warning
}

int *f5(void) {
  int *q = malloc(12);
  q = realloc(q, 20);
  return q; // no-warning
}

void f6(void) {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    free(p);
}

void f6_realloc(void) {
  int *p = malloc(12);
  if (!p)
    return; // no-warning
  else
    realloc(p,0);
}


char *doit2(void);
void pr6069(void) {
  char *buf = doit2();
  free(buf);
}

void pr6293(void) {
  free(0);
}

void f7(void) {
  char *x = (char*) malloc(4);
  free(x);
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
}

void f8(void) {
  char *x = (char*) malloc(4);
  free(x);
  char *y = strndup(x, 4); // expected-warning{{Use of memory after it is freed}}
}

void f7_realloc(void) {
  char *x = (char*) malloc(4);
  realloc(x,0);
  x[0] = 'a'; // expected-warning{{Use of memory after it is freed}}
}

void PR6123(void) {
  int *x = malloc(11); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
}

void PR7217(void) {
  int *buf = malloc(2); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  buf[1] = 'c'; // not crash
}

void cast_emtpy_struct(void) {
  struct st {
  };

  struct st *s = malloc(sizeof(struct st)); // no-warning
  free(s);
}

void cast_struct_1(void) {
  struct st {
    int i[100];
    char j[];
  };

  struct st *s = malloc(sizeof(struct st)); // no-warning
  free(s);
}

void cast_struct_2(void) {
  struct st {
    int i[100];
    char j[0];
  };

  struct st *s = malloc(sizeof(struct st)); // no-warning
  free(s);
}

void cast_struct_3(void) {
  struct st {
    int i[100];
    char j[1];
  };

  struct st *s = malloc(sizeof(struct st)); // no-warning
  free(s);
}

void cast_struct_4(void) {
  struct st {
    int i[100];
    char j[2];
  };

  struct st *s = malloc(sizeof(struct st)); // no-warning
  free(s);
}

void cast_struct_5(void) {
  struct st {
    char i[200];
    char j[1];
  };

  struct st *s = malloc(sizeof(struct st) - sizeof(char)); // no-warning
  free(s);
}

void cast_struct_warn_1(void) {
  struct st {
    int i[100];
    char j[2];
  };

  struct st *s = malloc(sizeof(struct st) + 2); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void cast_struct_warn_2(void) {
  struct st {
    int i[100];
    char j[2];
  };

  struct st *s = malloc(2); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void cast_struct_flex_array_1(void) {
  struct st {
    int i[100];
    char j[];
  };

  struct st *s = malloc(sizeof(struct st) + 3); // no-warning
  free(s);
}

void cast_struct_flex_array_2(void) {
  struct st {
    int i[100];
    char j[0];
  };

  struct st *s = malloc(sizeof(struct st) + 3); // no-warning
  free(s);
}

void cast_struct_flex_array_3(void) {
  struct st {
    int i[100];
    char j[1];
  };

  struct st *s = malloc(sizeof(struct st) + 3); // no-warning
  free(s);
}

void cast_struct_flex_array_4(void) {
  struct foo {
    char f[32];
  };
  struct st {
    char i[100];
    struct foo data[];
  };

  struct st *s = malloc(sizeof(struct st) + 3 * sizeof(struct foo)); // no-warning
  free(s);
}

void cast_struct_flex_array_5(void) {
  struct foo {
    char f[32];
  };
  struct st {
    char i[100];
    struct foo data[0];
  };

  struct st *s = malloc(sizeof(struct st) + 3 * sizeof(struct foo)); // no-warning
  free(s);
}

void cast_struct_flex_array_6(void) {
  struct foo {
    char f[32];
  };
  struct st {
    char i[100];
    struct foo data[1];
  };

  struct st *s = malloc(sizeof(struct st) + 3 * sizeof(struct foo)); // no-warning
  free(s);
}

void cast_struct_flex_array_warn_1(void) {
  struct foo {
    char f[32];
  };
  struct st {
    char i[100];
    struct foo data[];
  };

  struct st *s = malloc(3 * sizeof(struct st) + 3 * sizeof(struct foo)); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void cast_struct_flex_array_warn_2(void) {
  struct foo {
    char f[32];
  };
  struct st {
    char i[100];
    struct foo data[0];
  };

  struct st *s = malloc(3 * sizeof(struct st) + 3 * sizeof(struct foo)); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void cast_struct_flex_array_warn_3(void) {
  struct foo {
    char f[32];
  };
  struct st {
    char i[100];
    struct foo data[1];
  };

  struct st *s = malloc(3 * sizeof(struct st) + 3 * sizeof(struct foo)); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void cast_struct_flex_array_warn_4(void) {
  struct st {
    int i[100];
    int j[];
  };

  struct st *s = malloc(sizeof(struct st) + 3); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void cast_struct_flex_array_warn_5(void) {
  struct st {
    int i[100];
    int j[0];
  };

  struct st *s = malloc(sizeof(struct st) + 3); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void cast_struct_flex_array_warn_6(void) {
  struct st {
    int i[100];
    int j[1];
  };

  struct st *s = malloc(sizeof(struct st) + 3); // expected-warning{{Cast a region whose size is not a multiple of the destination type size}}
  free(s);
}

void mallocCastToVoid(void) {
  void *p = malloc(2);
  const void *cp = p; // not crash
  free(p);
}

void mallocCastToFP(void) {
  void *p = malloc(2);
  void (*fp)(void) = p; // not crash
  free(p);
}

// This tests that malloc() buffers are undefined by default
char mallocGarbage (void) {
	char *buf = malloc(2);
	char result = buf[1]; // expected-warning{{undefined}}
	free(buf);
	return result;
}

// This tests that calloc() buffers need to be freed
void callocNoFree (void) {
  char *buf = calloc(2,2);
  return; // expected-warning{{Potential leak of memory pointed to by 'buf'}}
}

// These test that calloc() buffers are zeroed by default
char callocZeroesGood (void) {
	char *buf = calloc(2,2);
	char result = buf[3]; // no-warning
	if (buf[1] == 0) {
	  free(buf);
	}
	return result; // no-warning
}

char callocZeroesBad (void) {
	char *buf = calloc(2,2);
	char result = buf[3]; // no-warning
	if (buf[1] != 0) {
	  free(buf); // expected-warning{{never executed}}
	}
	return result; // expected-warning{{Potential leak of memory pointed to by 'buf'}}
}

void nullFree(void) {
  int *p = 0;
  free(p); // no warning - a nop
}

void paramFree(int *p) {
  myfoo(p);
  free(p); // no warning
  myfoo(p); // expected-warning {{Use of memory after it is freed}}
}

int* mallocEscapeRet(void) {
  int *p = malloc(12);
  return p; // no warning
}

void mallocEscapeFoo(void) {
  int *p = malloc(12);
  myfoo(p);
  return; // no warning
}

void mallocEscapeFree(void) {
  int *p = malloc(12);
  myfoo(p);
  free(p);
}

void mallocEscapeFreeFree(void) {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  free(p); // expected-warning{{Attempt to free released memory}}
}

void mallocEscapeFreeUse(void) {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  myfoo(p); // expected-warning{{Use of memory after it is freed}}
}

int *myalloc(void);
void myalloc2(int **p);

void mallocEscapeFreeCustomAlloc(void) {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  p = myalloc();
  free(p); // no warning
}

void mallocEscapeFreeCustomAlloc2(void) {
  int *p = malloc(12);
  myfoo(p);
  free(p);
  myalloc2(&p);
  free(p); // no warning
}

void mallocBindFreeUse(void) {
  int *x = malloc(12);
  int *y = x;
  free(y);
  myfoo(x); // expected-warning{{Use of memory after it is freed}}
}

void mallocEscapeMalloc(void) {
  int *p = malloc(12);
  myfoo(p);
  p = malloc(12);
} // expected-warning{{Potential leak of memory pointed to by}}

void mallocMalloc(void) {
  int *p = malloc(12);
  p = malloc(12);
} // expected-warning {{Potential leak of memory pointed to by}}\
  // expected-warning {{Potential leak of memory pointed to by}}

void mallocFreeMalloc(void) {
  int *p = malloc(12);
  free(p);
  p = malloc(12);
  free(p);
}

void mallocFreeUse_params(void) {
  int *p = malloc(12);
  free(p);
  myfoo(p); //expected-warning{{Use of memory after it is freed}}
}

void mallocFreeUse_params2(void) {
  int *p = malloc(12);
  free(p);
  myfooint(*p); //expected-warning{{Use of memory after it is freed}}
}

void mallocFailedOrNot(void) {
  int *p = malloc(12);
  if (!p)
    free(p);
  else
    free(p);
}

struct StructWithInt {
  int g;
};

int *mallocReturnFreed(void) {
  int *p = malloc(12);
  free(p);
  return p; // expected-warning {{Use of memory after it is freed}}
}

int useAfterFreeStruct(void) {
  struct StructWithInt *px= malloc(sizeof(struct StructWithInt));
  px->g = 5;
  free(px);
  return px->g; // expected-warning {{Use of memory after it is freed}}
}

void nonSymbolAsFirstArg(int *pp, struct StructWithInt *p);

void mallocEscapeFooNonSymbolArg(void) {
  struct StructWithInt *p = malloc(sizeof(struct StructWithInt));
  nonSymbolAsFirstArg(&p->g, p);
  return; // no warning
}

void mallocFailedOrNotLeak(void) {
  int *p = malloc(12);
  if (p == 0)
    return; // no warning
  else
    return; // expected-warning {{Potential leak of memory pointed to by}}
}

void mallocAssignment(void) {
  char *p = malloc(12);
  p = fooRetPtr();
} // expected-warning {{leak}}

int vallocTest(void) {
  char *mem = valloc(12);
  return 0; // expected-warning {{Potential leak of memory pointed to by}}
}

void vallocEscapeFreeUse(void) {
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

void GlobalFree(void) {
  free(Gl);
}

void GlobalMalloc(void) {
  Gl = malloc(12);
}

void GlobalStructMalloc(void) {
  int *a = malloc(12);
  GlS.x = a;
}

void GlobalStructMallocFree(void) {
  int *a = malloc(12);
  GlS.x = a;
  free(GlS.x);
}

char *ArrayG[12];

void globalArrayTest(void) {
  char *p = (char*)malloc(12);
  ArrayG[0] = p;
}

// Make sure that we properly handle a pointer stored into a local struct/array.
typedef struct _StructWithPtr {
  int *memP;
} StructWithPtr;

static StructWithPtr arrOfStructs[10];

void testMalloc(void) {
  int *x = malloc(12);
  StructWithPtr St;
  St.memP = x;
  arrOfStructs[0] = St; // no-warning
}

StructWithPtr testMalloc2(void) {
  int *x = malloc(12);
  StructWithPtr St;
  St.memP = x;
  return St; // no-warning
}

int *testMalloc3(void) {
  int *x = malloc(12);
  int *y = x;
  return y; // no-warning
}

void testStructLeak(void) {
  StructWithPtr St;
  St.memP = malloc(12);
  return; // expected-warning {{Potential leak of memory pointed to by 'St.memP'}}
}

void testElemRegion1(void) {
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
  return px; // expected-warning {{Potential leak of memory pointed to by}}
}

struct X* RegInvalidationGiveUp1(void) {
  int *p = malloc(12);
  struct X *px= malloc(sizeof(struct X));
  px->p = p;
  return px;
}

int **RegInvalidationDetect2(int **pp) {
  int *p = malloc(12);
  pp = &p;
  pp++;
  return 0;// expected-warning {{Potential leak of memory pointed to by}}
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
  strcpy(p, s);
  strcpy(s, p);
  strcpy(p, p);
  memcpy(p, s, 1);
  memcpy(s, p, 1);
  memcpy(p, p, 1);
} // expected-warning {{leak}}

// Treat source buffer contents as escaped.
void escapeSourceContents(char *s) {
  char *p = malloc(12);
  memcpy(s, &p, 12); // no warning

  void *p1 = malloc(7);
  char *a;
  memcpy(&a, &p1, sizeof a);
  // FIXME: No warning due to limitations imposed by current modelling of
  // 'memcpy' (regions metadata is not copied).

  int *ptrs[2];
  int *allocated = (int *)malloc(4);
  memcpy(&ptrs[0], &allocated, sizeof(int *));
  // FIXME: No warning due to limitations imposed by current modelling of
  // 'memcpy' (regions metadata is not copied).
}

void invalidateDestinationContents(void) {
  int *null = 0;
  int *p = (int *)malloc(4);
  memcpy(&p, &null, sizeof(int *));

  int *ptrs1[2]; // expected-warning {{Potential leak of memory pointed to by}}
  ptrs1[0] = (int *)malloc(4);
  memcpy(ptrs1,  &null, sizeof(int *));

  int *ptrs2[2]; // expected-warning {{Potential memory leak}}
  ptrs2[0] = (int *)malloc(4);
  memcpy(&ptrs2[1],  &null, sizeof(int *));

  int *ptrs3[2]; // expected-warning {{Potential memory leak}}
  ptrs3[0] = (int *)malloc(4);
  memcpy(&ptrs3[0],  &null, sizeof(int *));
} // expected-warning {{Potential memory leak}}

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
int *specialMallocWithStruct(void) {
  struct StructWithInt *px= malloc(sizeof(struct StructWithInt));
  return &(px->g);
}

// Test various allocation/deallocation functions.
void testStrdup(const char *s, unsigned validIndex) {
  char *s2 = strdup(s);
  s2[validIndex + 1] = 'b';
} // expected-warning {{Potential leak of memory pointed to by}}

void testWinStrdup(const char *s, unsigned validIndex) {
  char *s2 = _strdup(s);
  s2[validIndex + 1] = 'b';
} // expected-warning {{Potential leak of memory pointed to by}}

void testWcsdup(const wchar_t *s, unsigned validIndex) {
  wchar_t *s2 = wcsdup(s);
  s2[validIndex + 1] = 'b';
} // expected-warning {{Potential leak of memory pointed to by}}

void testWinWcsdup(const wchar_t *s, unsigned validIndex) {
  wchar_t *s2 = _wcsdup(s);
  s2[validIndex + 1] = 'b';
} // expected-warning {{Potential leak of memory pointed to by}}

int testStrndup(const char *s, unsigned validIndex, unsigned size) {
  char *s2 = strndup(s, size);
  s2 [validIndex + 1] = 'b';
  if (s2[validIndex] != 'a')
    return 0;
  else
    return 1;// expected-warning {{Potential leak of memory pointed to by}}
}

void testStrdupContentIsDefined(const char *s, unsigned validIndex) {
  char *s2 = strdup(s);
  char result = s2[1];// no warning
  free(s2);
}

void testWinStrdupContentIsDefined(const char *s, unsigned validIndex) {
  char *s2 = _strdup(s);
  char result = s2[1];// no warning
  free(s2);
}

void testWcsdupContentIsDefined(const wchar_t *s, unsigned validIndex) {
  wchar_t *s2 = wcsdup(s);
  wchar_t result = s2[1];// no warning
  free(s2);
}

void testWinWcsdupContentIsDefined(const wchar_t *s, unsigned validIndex) {
  wchar_t *s2 = _wcsdup(s);
  wchar_t result = s2[1];// no warning
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
FILE *useFunOpen(void) {
    void *ctx = malloc(sizeof(int));
    FILE *f = funopen(ctx, 0, 0, 0, releasePtr); // no warning
    if (f == 0) {
        free(ctx);
    }
    return f;
}
FILE *useFunOpenNoReleaseFunction(void) {
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
FILE *useFunOpenReadNoRelease(void) {
  void *ctx = malloc(sizeof(int));
  FILE *f = funopen(ctx, readNothing, 0, 0, 0);
  if (f == 0) {
    free(ctx);
  }
  return f; // expected-warning{{leak}}
}

// Test setbuf, setvbuf.
int my_main_no_warning(void) {
    char *p = malloc(100);
    setvbuf(stdout, p, 0, 100);
    return 0;
}
int my_main_no_warning2(void) {
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
  if (buffer == stackBuffer)
    return;
  else
    return; // expected-warning {{leak}}
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

void radar_11358224_test_double_assign_ints_positive_2(void)
{
  void *ptr = malloc(16);
  ptr = ptr;
} // expected-warning {{leak}}

// Assume that functions which take a function pointer can free memory even if
// they are defined in system headers and take the const pointer to the
// allocated memory. (radar://11160612)
int const_ptr_and_callback(int, const char*, int n, void(*)(void*));
void r11160612_1(void) {
  char *x = malloc(12);
  const_ptr_and_callback(0, x, 12, free); // no - warning
}

// Null is passed as callback.
void r11160612_2(void) {
  char *x = malloc(12);
  const_ptr_and_callback(0, x, 12, 0);
} // expected-warning {{leak}}

// Callback is passed to a function defined in a system header.
void r11160612_4(void) {
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

int CMPRegionHeapToStack(void) {
  int x = 0;
  int *x1 = malloc(8);
  int *x2 = &x;
  clang_analyzer_eval(x1 == x2); // expected-warning{{FALSE}}
  free(x1);
  return x;
}

int CMPRegionHeapToHeap2(void) {
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

int CMPRegionHeapToHeap(void) {
  int x = 0;
  int *x1 = malloc(8);
  int *x4 = x1;
  if (x1 == x4) {
    free(x1);
    return 5/x; // expected-warning{{Division by zero}}
  }
  return x;// expected-warning{{This statement is never executed}}
}

int HeapAssignment(void) {
  int m = 0;
  int *x = malloc(4);
  int *y = x;
  *x = 5;
  clang_analyzer_eval(*x != *y); // expected-warning{{FALSE}}
  free(x);
  return 0;
}

int *retPtr(void);
int *retPtrMightAlias(int *x);
int cmpHeapAllocationToUnknown(void) {
  int zero = 0;
  int *yBefore = retPtr();
  int *m = malloc(8);
  int *yAfter = retPtrMightAlias(m);
  clang_analyzer_eval(yBefore == m); // expected-warning{{FALSE}}
  clang_analyzer_eval(yAfter == m); // expected-warning{{FALSE}}
  free(m);
  return 0;
}

void localArrayTest(void) {
  char *p = (char*)malloc(12);
  char *ArrayL[12];
  ArrayL[0] = p;
} // expected-warning {{leak}}

void localStructTest(void) {
  StructWithPtr St;
  StructWithPtr *pSt = &St;
  pSt->memP = malloc(12);
} // expected-warning{{Potential leak of memory pointed to by}}

#ifdef __INTPTR_TYPE__
// Test double assignment through integers.
typedef __INTPTR_TYPE__ intptr_t;
typedef unsigned __INTPTR_TYPE__ uintptr_t;

static intptr_t glob;
void test_double_assign_ints(void)
{
  void *ptr = malloc (16);  // no-warning
  glob = (intptr_t)(uintptr_t)ptr;
}

void test_double_assign_ints_positive(void)
{
  void *ptr = malloc(16);
  (void*)(intptr_t)(uintptr_t)ptr; // expected-warning {{unused}}
} // expected-warning {{leak}}
#endif

void testCGContextNoLeak(void)
{
  void *ptr = malloc(16);
  CGContextRef context = CGBitmapContextCreate(ptr);

  // Because you can get the data back out like this, even much later,
  // CGBitmapContextCreate is one of our "stop-tracking" exceptions.
  free(CGBitmapContextGetData(context));
}

void testCGContextLeak(void)
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
  // We don't expect a use-after-free for a->P here because the warning above
  // is a sink.
  return a->p; // no-warning
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


char *testLeakWithinReturn(char *str) {
  return strdup(strdup(str)); // expected-warning{{leak}}
}

char *testWinLeakWithinReturn(char *str) {
  return _strdup(_strdup(str)); // expected-warning{{leak}}
}

wchar_t *testWinWideLeakWithinReturn(wchar_t *str) {
  return _wcsdup(_wcsdup(str)); // expected-warning{{leak}}
}

void passConstPtr(const char * ptr);

void testPassConstPointer(void) {
  char * string = malloc(sizeof(char)*10);
  passConstPtr(string);
  return; // expected-warning {{leak}}
}

void testPassConstPointerIndirectly(void) {
  char *p = malloc(1);
  p++;
  memcmp(p, p, sizeof(&p));
  return; // expected-warning {{leak}}
}

void testPassConstPointerIndirectlyStruct(void) {
  struct HasPtr hp;
  hp.p = malloc(10);
  memcmp(&hp, &hp, sizeof(hp));
  return; // expected-warning {{Potential leak of memory pointed to by 'hp.p'}}
}

void testPassToSystemHeaderFunctionIndirectlyStruct(void) {
  SomeStruct ss;
  ss.p = malloc(1);
  fakeSystemHeaderCall(&ss); // invalidates ss, making ss.p unreachable
  // Technically a false negative here -- we know the system function won't free
  // ss.p, but nothing else will either!
} // no-warning

void testPassToSystemHeaderFunctionIndirectlyStructFree(void) {
  SomeStruct ss;
  ss.p = malloc(1);
  fakeSystemHeaderCall(&ss); // invalidates ss, making ss.p unreachable
  free(ss.p);
} // no-warning

void testPassToSystemHeaderFunctionIndirectlyArray(void) {
  int *p[1];
  p[0] = malloc(sizeof(int));
  fakeSystemHeaderCallIntPtr(p); // invalidates p, making p[0] unreachable
  // Technically a false negative here -- we know the system function won't free
  // p[0], but nothing else will either!
} // no-warning

void testPassToSystemHeaderFunctionIndirectlyArrayFree(void) {
  int *p[1];
  p[0] = malloc(sizeof(int));
  fakeSystemHeaderCallIntPtr(p); // invalidates p, making p[0] unreachable
  free(p[0]);
} // no-warning

int *testOffsetAllocate(size_t size) {
  int *memoryBlock = (int *)malloc(size + sizeof(int));
  return &memoryBlock[1]; // no-warning
}

void testOffsetDeallocate(int *memoryBlock) {
  free(&memoryBlock[-1]);  // no-warning
}

void testOffsetOfRegionFreed(void) {
  __int64_t * array = malloc(sizeof(__int64_t)*2);
  array += 1;
  free(&array[0]); // expected-warning{{Argument to free() is offset by 8 bytes from the start of memory allocated by malloc()}}
}

void testOffsetOfRegionFreed2(void) {
  __int64_t *p = malloc(sizeof(__int64_t)*2);
  p += 1;
  free(p); // expected-warning{{Argument to free() is offset by 8 bytes from the start of memory allocated by malloc()}}
}

void testOffsetOfRegionFreed3(void) {
  char *r = malloc(sizeof(char));
  r = r - 10;
  free(r); // expected-warning {{Argument to free() is offset by -10 bytes from the start of memory allocated by malloc()}}
}

void testOffsetOfRegionFreedAfterFunctionCall(void) {
  int *p = malloc(sizeof(int)*2);
  p += 1;
  myfoo(p);
  free(p); // expected-warning{{Argument to free() is offset by 4 bytes from the start of memory allocated by malloc()}}
}

void testFixManipulatedPointerBeforeFree(void) {
  int * array = malloc(sizeof(int)*2);
  array += 1;
  free(&array[-1]); // no-warning
}

void testFixManipulatedPointerBeforeFree2(void) {
  char *r = malloc(sizeof(char));
  r = r + 10;
  free(r-10); // no-warning
}

void freeOffsetPointerPassedToFunction(void) {
  __int64_t *p = malloc(sizeof(__int64_t)*2);
  p[1] = 0;
  p += 1;
  myfooint(*p); // not passing the pointer, only a value pointed by pointer
  free(p); // expected-warning {{Argument to free() is offset by 8 bytes from the start of memory allocated by malloc()}}
}

int arbitraryInt(void);
void freeUnknownOffsetPointer(void) {
  char *r = malloc(sizeof(char));
  r = r + arbitraryInt(); // unable to reason about what the offset might be
  free(r); // no-warning
}

void testFreeNonMallocPointerWithNoOffset(void) {
  char c;
  char *r = &c;
  r = r + 10;
  free(r-10); // expected-warning {{Argument to free() is the address of the local variable 'c', which is not memory allocated by malloc()}}
}

void testFreeNonMallocPointerWithOffset(void) {
  char c;
  char *r = &c;
  free(r+1); // expected-warning {{Argument to free() is the address of the local variable 'c', which is not memory allocated by malloc()}}
}

void testOffsetZeroDoubleFree(void) {
  int *array = malloc(sizeof(int)*2);
  int *p = &array[0];
  free(p);
  free(&array[0]); // expected-warning{{Attempt to free released memory}}
}

void testOffsetPassedToStrlen(void) {
  char * string = malloc(sizeof(char)*10);
  string += 1;
  int length = strlen(string); // expected-warning {{Potential leak of memory pointed to by 'string'}}
}

void testOffsetPassedToStrlenThenFree(void) {
  char * string = malloc(sizeof(char)*10);
  string += 1;
  int length = strlen(string);
  free(string); // expected-warning {{Argument to free() is offset by 1 byte from the start of memory allocated by malloc()}}
}

void testOffsetPassedAsConst(void) {
  char * string = malloc(sizeof(char)*10);
  string += 1;
  passConstPtr(string);
  free(string); // expected-warning {{Argument to free() is offset by 1 byte from the start of memory allocated by malloc()}}
}

char **_vectorSegments;
int _nVectorSegments;

void poolFreeC(void* s) {
  free(s); // no-warning
}
void freeMemory(void) {
  while (_nVectorSegments) {
    poolFreeC(_vectorSegments[_nVectorSegments++]);
  }
}

// PR16730
void testReallocEscaped(void **memory) {
  *memory = malloc(47);
  char *new_memory = realloc(*memory, 47);
  if (new_memory != 0) {
    *memory = new_memory;
  }
}

// PR16558
void *smallocNoWarn(size_t size) {
  if (size == 0) {
    return malloc(1); // this branch is never called
  }
  else {
    return malloc(size);
  }
}

char *dupstrNoWarn(const char *s) {
  const int len = strlen(s);
  char *p = (char*) smallocNoWarn(len + 1);
  strcpy(p, s); // no-warning
  return p;
}

void *smallocWarn(size_t size) {
  if (size == 2) {
    return malloc(1);
  }
  else {
    return malloc(size);
  }
}

int *radar15580979(void) {
  int *data = (int *)malloc(32);
  int *p = data ?: (int*)malloc(32); // no warning
  return p;
}

// Some data structures may hold onto the pointer and free it later.
void testEscapeThroughSystemCallTakingVoidPointer1(void *queue) {
  int *data = (int *)malloc(32);
  fake_insque(queue, data); // no warning
}

void testEscapeThroughSystemCallTakingVoidPointer2(fake_rb_tree_t *rbt) {
  int *data = (int *)malloc(32);
  fake_rb_tree_init(rbt, data);
} //expected-warning{{Potential leak}}

void testEscapeThroughSystemCallTakingVoidPointer3(fake_rb_tree_t *rbt) {
  int *data = (int *)malloc(32);
  fake_rb_tree_init(rbt, data);
  fake_rb_tree_insert_node(rbt, data); // no warning
}

struct IntAndPtr {
  int x;
  int *p;
};

void constEscape(const void *ptr);

void testConstEscapeThroughAnotherField(void) {
  struct IntAndPtr s;
  s.p = malloc(sizeof(int));
  constEscape(&(s.x)); // could free s->p!
} // no-warning

// PR15623
int testNoCheckerDataPropogationFromLogicalOpOperandToOpResult(void) {
   char *param = malloc(10);
   char *value = malloc(10);
   int ok = (param && value);
   free(param);
   free(value);
   // Previously we ended up with 'Use of memory after it is freed' on return.
   return ok; // no warning
}

void (*fnptr)(int);
void freeIndirectFunctionPtr(void) {
  void *p = (void *)fnptr;
  free(p); // expected-warning {{Argument to free() is a function pointer}}
}

void freeFunctionPtr(void) {
  free((void *)fnptr);
  // expected-warning@-1{{Argument to free() is a function pointer}}
  // expected-warning@-2{{attempt to call free on non-heap object '(void *)fnptr'}}
}

void allocateSomeMemory(void *offendingParameter, void **ptr) {
  *ptr = malloc(1);
}

void testNoCrashOnOffendingParameter(void) {
  // "extern" is necessary to avoid unrelated warnings
  // on passing uninitialized value.
  extern void *offendingParameter;
  void* ptr;
  allocateSomeMemory(offendingParameter, &ptr);
} // expected-warning {{Potential leak of memory pointed to by 'ptr'}}


// Test a false positive caused by a bug in liveness analysis.
struct A {
  int *buf;
};
struct B {
  struct A *a;
};
void livenessBugRealloc(struct A *a) {
  a->buf = realloc(a->buf, sizeof(int)); // no-warning
}
void testLivenessBug(struct B *in_b) {
  struct B *b = in_b;
  livenessBugRealloc(b->a);
 ((void) 0); // An attempt to trick liveness analysis.
  livenessBugRealloc(b->a);
}

struct ListInfo {
  struct ListInfo *next;
};

struct ConcreteListItem {
  struct ListInfo li;
  int i;
};

void list_add(struct ListInfo *list, struct ListInfo *item);

void testCStyleListItems(struct ListInfo *list) {
  struct ConcreteListItem *x = malloc(sizeof(struct ConcreteListItem));
  list_add(list, &x->li); // will free 'x'.
}

// MEM34-C. Only free memory allocated dynamically
// Second non-compliant example.
// https://wiki.sei.cmu.edu/confluence/display/c/MEM34-C.+Only+free+memory+allocated+dynamically
enum { BUFSIZE = 256 };

void MEM34_C(void) {
  char buf[BUFSIZE];
  char *p = (char *)realloc(buf, 2 * BUFSIZE);
  // expected-warning@-1{{Argument to realloc() is the address of the local \
variable 'buf', which is not memory allocated by malloc() [unix.Malloc]}}
  if (p == NULL) {
    /* Handle error */
  }
}

(*crash_a)(); // expected-warning{{type specifier missing}}
// A CallEvent without a corresponding FunctionDecl.
crash_b() { crash_a(); } // no-crash
// expected-warning@-1{{type specifier missing}} expected-warning@-1{{non-void}}

long *global_a;
void realloc_crash(void) {
  long *c = global_a;
  c--;
  realloc(c, 8); // no-crash
} // expected-warning{{Potential memory leak [unix.Malloc]}}

// ----------------------------------------------------------------------------
// False negatives.

void testMallocWithParam(int **p) {
  *p = (int*) malloc(sizeof(int));
  *p = 0; // FIXME: should warn here
}

void testMallocWithParam_2(int **p) {
  *p = (int*) malloc(sizeof(int)); // no-warning
}

void testPassToSystemHeaderFunctionIndirectly(void) {
  int *p = malloc(4);
  p++;
  fakeSystemHeaderCallInt(p);
  // FIXME: This is a leak: if we think a system function won't free p, it
  // won't free (p-1) either.
}

void testMallocIntoMalloc(void) {
  StructWithPtr *s = malloc(sizeof(StructWithPtr));
  s->memP = malloc(sizeof(int));
  free(s);
} // FIXME: should warn here

int conjure(void);
void testExtent(void) {
  int x = conjure();
  clang_analyzer_dump(x);
  // expected-warning-re@-1 {{{{^conj_\$[[:digit:]]+{int, LC1, S[[:digit:]]+, #1}}}}}}
  int *p = (int *)malloc(x);
  clang_analyzer_dumpExtent(p);
  // expected-warning-re@-1 {{{{^conj_\$[[:digit:]]+{int, LC1, S[[:digit:]]+, #1}}}}}}
  free(p);
}
