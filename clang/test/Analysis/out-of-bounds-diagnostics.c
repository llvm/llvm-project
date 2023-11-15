// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-output=text        \
// RUN:     -analyzer-checker=core,alpha.security.ArrayBoundV2,unix.Malloc,alpha.security.taint -verify %s

int array[10];

void arrayUnderflow(void) {
  array[-3] = 5;
  // expected-warning@-1 {{Out of bound access to memory preceding 'array'}}
  // expected-note@-2 {{Access of 'array' at negative byte offset -12}}
}

int scanf(const char *restrict fmt, ...);

void taintedIndex(void) {
  int index;
  scanf("%d", &index);
  // expected-note@-1 {{Taint originated here}}
  // expected-note@-2 {{Taint propagated to the 2nd argument}}
  array[index] = 5;
  // expected-warning@-1 {{Potential out of bound access to 'array' with tainted offset}}
  // expected-note@-2 {{Access of 'array' with a tainted offset that may be too large}}
}

void arrayOverflow(void) {
  array[12] = 5;
  // expected-warning@-1 {{Out of bound access to memory after the end of 'array'}}
  // expected-note@-2 {{Access of 'array' at index 12, while it holds only 10 'int' elements}}
}

int scalar;
int scalarOverflow(void) {
  return (&scalar)[1];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'scalar'}}
  // expected-note@-2 {{Access of 'scalar' at index 1, while it holds only a single 'int' element}}
}

int oneElementArray[1];
int oneElementArrayOverflow(void) {
  return oneElementArray[1];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'oneElementArray'}}
  // expected-note@-2 {{Access of 'oneElementArray' at index 1, while it holds only a single 'int' element}}
}

short convertedArray(void) {
  return ((short*)array)[47];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'array'}}
  // expected-note@-2 {{Access of 'array' at index 47, while it holds only 20 'short' elements}}
}

struct vec {
  int len;
  double elems[64];
} v;

double arrayInStruct(void) {
  return v.elems[64];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'v.elems'}}
  // expected-note@-2 {{Access of 'v.elems' at index 64, while it holds only 64 'double' elements}}
}

double arrayInStructPtr(struct vec *pv) {
  return pv->elems[64];
  // expected-warning@-1 {{Out of bound access to memory after the end of the field 'elems'}}
  // expected-note@-2 {{Access of the field 'elems' at index 64, while it holds only 64 'double' elements}}
}

struct two_bytes {
  char lo, hi;
};

struct two_bytes convertedArray2(void) {
  // We report this with byte offsets because the offset is not divisible by the element size.
  struct two_bytes a = {0, 0};
  char *p = (char*)&a;
  return *((struct two_bytes*)(p + 7));
  // expected-warning@-1 {{Out of bound access to memory after the end of 'a'}}
  // expected-note@-2 {{Access of 'a' at byte offset 7, while it holds only 2 bytes}}
}

int intFromString(void) {
  // We report this with byte offsets because the extent is not divisible by the element size.
  return ((const int*)"this is a string of 33 characters")[20];
  // expected-warning@-1 {{Out of bound access to memory after the end of the string literal}}
  // expected-note@-2 {{Access of the string literal at byte offset 80, while it holds only 34 bytes}}
}

int intFromStringDivisible(void) {
  // However, this is reported with indices/elements, because the extent happens to be a multiple of 4.
  return ((const int*)"abc")[20];
  // expected-warning@-1 {{Out of bound access to memory after the end of the string literal}}
  // expected-note@-2 {{Access of the string literal at index 20, while it holds only a single 'int' element}}
}

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t size);

int *mallocRegion(void) {
  int *mem = (int*)malloc(2*sizeof(int));
  mem[3] = -2;
  // expected-warning@-1 {{Out of bound access to memory after the end of the heap area}}
  // expected-note@-2 {{Access of the heap area at index 3, while it holds only 2 'int' elements}}
  return mem;
}

void *alloca(size_t size);

int allocaRegion(void) {
  int *mem = (int*)alloca(2*sizeof(int));
  mem[3] = -2;
  // expected-warning@-1 {{Out of bound access to memory after the end of the memory returned by 'alloca'}}
  // expected-note@-2 {{Access of the memory returned by 'alloca' at index 3, while it holds only 2 'int' elements}}
  return *mem;
}

int *unknownExtent(int arg) {
  if (arg >= 2)
    return 0;
  int *mem = (int*)malloc(arg);
  mem[8] = -2;
  // FIXME: this should produce
  //   {{Out of bound access to memory after the end of the heap area}}
  //   {{Access of 'int' element in the heap area at index 8}}
  return mem;
}

void unknownIndex(int arg) {
  // expected-note@+2 {{Assuming 'arg' is >= 12}}
  // expected-note@+1 {{Taking true branch}}
  if (arg >= 12)
    array[arg] = -2;
  // expected-warning@-1 {{Out of bound access to memory after the end of 'array'}}
  // expected-note@-2 {{Access of 'array' at an overflowing index, while it holds only 10 'int' elements}}
}

int *nothingIsCertain(int x, int y) {
  if (x >= 2)
    return 0;
  int *mem = (int*)malloc(x);
  if (y >= 8)
    mem[y] = -2;
  // FIXME: this should produce
  //   {{Out of bound access to memory after the end of the heap area}}
  //   {{Access of 'int' element in the heap area at an overflowing index}}
  return mem;
}
