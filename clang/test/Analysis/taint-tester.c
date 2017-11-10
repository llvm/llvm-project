// RUN: %clang_analyze_cc1 -Wno-int-to-pointer-cast -analyzer-checker=alpha.security.taint,debug.TaintTest %s -verify

#include "Inputs/system-header-simulator.h"

#define BUFSIZE 10
int Buffer[BUFSIZE];

struct XYStruct {
  int x;
  int y;
  char z;
};

void taintTracking(int x) {
  int n;
  int *addr = &Buffer[0];
  scanf("%d", &n);
  addr += n;// expected-warning + {{tainted}}
  *addr = n; // expected-warning + {{tainted}}

  double tdiv = n / 30; // expected-warning+ {{tainted}}
  char *loc_cast = (char *) n; // expected-warning +{{tainted}}
  char tinc = tdiv++; // expected-warning + {{tainted}}
  int tincdec = (char)tinc--; // expected-warning+{{tainted}}

  // Tainted ptr arithmetic/array element address.
  int tprtarithmetic1 = *(addr+1); // expected-warning + {{tainted}}

  // Dereference.
  int *ptr;
  scanf("%p", &ptr);
  int ptrDeref = *ptr; // expected-warning + {{tainted}}
  int _ptrDeref = ptrDeref + 13; // expected-warning + {{tainted}}

  // Pointer arithmetic + dereferencing.
  // FIXME: We fail to propagate the taint here because RegionStore does not
  // handle ElementRegions with symbolic indexes.
  int addrDeref = *addr; // expected-warning + {{tainted}}
  int _addrDeref = addrDeref; // expected-warning + {{tainted}}

  // Tainted struct address, casts.
  struct XYStruct *xyPtr = 0;
  scanf("%p", &xyPtr);
  void *tXYStructPtr = xyPtr; // expected-warning + {{tainted}}
  struct XYStruct *xyPtrCopy = tXYStructPtr; // expected-warning + {{tainted}}
  int ptrtx = xyPtr->x;// expected-warning + {{tainted}}
  int ptrty = xyPtr->y;// expected-warning + {{tainted}}

  // Taint on fields of a struct.
  struct XYStruct xy = {2, 3, 11};
  scanf("%d", &xy.y);
  scanf("%d", &xy.x);
  int tx = xy.x; // expected-warning + {{tainted}}
  int ty = xy.y; // FIXME: This should be tainted as well.
  char ntz = xy.z;// no warning
  // Now, scanf scans both.
  scanf("%d %d", &xy.y, &xy.x);
  int ttx = xy.x; // expected-warning + {{tainted}}
  int tty = xy.y; // expected-warning + {{tainted}}
}

void BitwiseOp(int in, char inn) {
  // Taint on bitwise operations, integer to integer cast.
  int m;
  int x = 0;
  scanf("%d", &x);
  int y = (in << (x << in)) * 5;// expected-warning + {{tainted}}
  // The next line tests integer to integer cast.
  int z = y & inn; // expected-warning + {{tainted}}
  if (y == 5) // expected-warning + {{tainted}}
    m = z | z;// expected-warning + {{tainted}}
  else
    m = inn;
  int mm = m; // expected-warning + {{tainted}}
}

// Test getenv.
char *getenv(const char *name);
void getenvTest(char *home) {
  home = getenv("HOME"); // expected-warning + {{tainted}}
  if (home != 0) { // expected-warning + {{tainted}}
      char d = home[0]; // expected-warning + {{tainted}}
    }
}

int fscanfTest(void) {
  FILE *fp;
  char s[80];
  int t;

  // Check if stdin is treated as tainted.
  fscanf(stdin, "%s %d", s, &t);
  // Note, here, s is not tainted, but the data s points to is tainted.
  char *ts = s;
  char tss = s[0]; // expected-warning + {{tainted}}
  int tt = t; // expected-warning + {{tainted}}
  if((fp=fopen("test", "w")) == 0) // expected-warning + {{tainted}}
    return 1;
  fprintf(fp, "%s %d", s, t); // expected-warning + {{tainted}}
  fclose(fp); // expected-warning + {{tainted}}

  // Test fscanf and fopen.
  if((fp=fopen("test","r")) == 0) // expected-warning + {{tainted}}
    return 1;
  fscanf(fp, "%s%d", s, &t); // expected-warning + {{tainted}}
  fprintf(stdout, "%s %d", s, t); // expected-warning + {{tainted}}
  return 0;
}

// Check if we propagate taint from stdin when it's used in an assignment.
void stdinTest1() {
  int i;
  fscanf(stdin, "%d", &i);
  int j = i; // expected-warning + {{tainted}}
}
void stdinTest2(FILE *pIn) {
  FILE *p = stdin;
  FILE *pp = p;
  int ii;

  fscanf(pp, "%d", &ii);
  int jj = ii;// expected-warning + {{tainted}}

  fscanf(p, "%d", &ii);
  int jj2 = ii;// expected-warning + {{tainted}}

  ii = 3;
  int jj3 = ii;// no warning

  p = pIn;
  fscanf(p, "%d", &ii);
  int jj4 = ii;// no warning
}

void stdinTest3() {
  FILE **ppp = &stdin;
  int iii;
  fscanf(*ppp, "%d", &iii);
  int jjj = iii;// expected-warning + {{tainted}}
}

// Test that stdin does not get invalidated by calls.
void foo();
void stdinTest4() {
  int i;
  fscanf(stdin, "%d", &i);
  foo();
  int j = i; // expected-warning + {{tainted}}
}

int getw(FILE *);
void getwTest() {
  int i = getw(stdin); // expected-warning + {{tainted}}
}

typedef long ssize_t;
ssize_t getline(char ** __restrict, size_t * __restrict, FILE * __restrict);
int  printf(const char * __restrict, ...);
void free(void *ptr);
void getlineTest(void) {
  FILE *fp;
  char *line = 0;
  size_t len = 0;
  ssize_t read;
  while ((read = getline(&line, &len, stdin)) != -1) {
    printf("%s", line); // expected-warning + {{tainted}}
  }
  free(line); // expected-warning + {{tainted}}
}

// Test propagation functions - the ones that propagate taint from arguments to
// return value, ptr arguments.

int atoi(const char *nptr);
long atol(const char *nptr);
long long atoll(const char *nptr);

void atoiTest() {
  char s[80];
  scanf("%s", s);
  int d = atoi(s); // expected-warning + {{tainted}}
  int td = d; // expected-warning + {{tainted}}

  long l = atol(s); // expected-warning + {{tainted}}
  int tl = l; // expected-warning + {{tainted}}

  long long ll = atoll(s); // expected-warning + {{tainted}}
  int tll = ll; // expected-warning + {{tainted}}

}

char *pointer1;
void *pointer2;
void noCrashTest() {
  if (!*pointer1) {
    __builtin___memcpy_chk(pointer2, pointer1, 0, 0); // no-crash
  }
}
