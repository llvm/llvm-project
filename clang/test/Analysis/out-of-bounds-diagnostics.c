// RUN: %clang_analyze_cc1 -Wno-array-bounds -analyzer-output=text        \
// RUN:     -analyzer-checker=core,alpha.security.ArrayBoundV2,unix.Malloc,optin.taint -verify %s

int TenElements[10];

void arrayUnderflow(void) {
  TenElements[-3] = 5;
  // expected-warning@-1 {{Out of bound access to memory preceding 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at negative byte offset -12}}
}

int underflowWithDeref(void) {
  int *p = TenElements;
  --p;
  return *p;
  // expected-warning@-1 {{Out of bound access to memory preceding 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at negative byte offset -4}}
}

int rng(void);
int getIndex(void) {
  switch (rng()) {
    case 1: return -152;
    case 2: return -160;
    case 3: return -168;
    default: return -172;
  }
}

void gh86959(void) {
  // Previously code like this produced many almost-identical bug reports that
  // only differed in the offset value. Verify that now we only see one report.

  // expected-note@+1 {{Entering loop body}}
  while (rng())
    TenElements[getIndex()] = 10;
  // expected-warning@-1 {{Out of bound access to memory preceding 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at negative byte offset -688}}
}

int scanf(const char *restrict fmt, ...);

void taintedIndex(void) {
  int index;
  scanf("%d", &index);
  // expected-note@-1 {{Taint originated here}}
  // expected-note@-2 {{Taint propagated to the 2nd argument}}
  TenElements[index] = 5;
  // expected-warning@-1 {{Potential out of bound access to 'TenElements' with tainted index}}
  // expected-note@-2 {{Access of 'TenElements' with a tainted index that may be negative or too large}}
}

void taintedIndexNonneg(void) {
  int index;
  scanf("%d", &index);
  // expected-note@-1 {{Taint originated here}}
  // expected-note@-2 {{Taint propagated to the 2nd argument}}

  // expected-note@+2 {{Assuming 'index' is >= 0}}
  // expected-note@+1 {{Taking false branch}}
  if (index < 0)
    return;

  TenElements[index] = 5;
  // expected-warning@-1 {{Potential out of bound access to 'TenElements' with tainted index}}
  // expected-note@-2 {{Access of 'TenElements' with a tainted index that may be too large}}
}

void taintedIndexUnsigned(void) {
  unsigned index;
  scanf("%u", &index);
  // expected-note@-1 {{Taint originated here}}
  // expected-note@-2 {{Taint propagated to the 2nd argument}}

  TenElements[index] = 5;
  // expected-warning@-1 {{Potential out of bound access to 'TenElements' with tainted index}}
  // expected-note@-2 {{Access of 'TenElements' with a tainted index that may be too large}}
}

int *taintedIndexAfterTheEndPtr(void) {
  // NOTE: Technically speaking, this testcase does not trigger any UB because
  // &TenElements[10] is the after-the-end pointer which is well-defined; but
  // this is a bug-prone situation and far from the idiomatic use of
  // `&TenElements[size]`, so it's better to report an error. This report can
  // be easily silenced by writing TenElements+index instead of
  // &TenElements[index].
  int index;
  scanf("%d", &index);
  // expected-note@-1 {{Taint originated here}}
  // expected-note@-2 {{Taint propagated to the 2nd argument}}
  if (index < 0 || index > 10)
    return TenElements;
  // expected-note@-2 {{Assuming 'index' is >= 0}}
  // expected-note@-3 {{Left side of '||' is false}}
  // expected-note@-4 {{Assuming 'index' is <= 10}}
  // expected-note@-5 {{Taking false branch}}
  return &TenElements[index];
  // expected-warning@-1 {{Potential out of bound access to 'TenElements' with tainted index}}
  // expected-note@-2 {{Access of 'TenElements' with a tainted index that may be too large}}
}

void taintedOffset(void) {
  int index;
  scanf("%d", &index);
  // expected-note@-1 {{Taint originated here}}
  // expected-note@-2 {{Taint propagated to the 2nd argument}}
  int *p = TenElements + index;
  p[0] = 5;
  // expected-warning@-1 {{Potential out of bound access to 'TenElements' with tainted offset}}
  // expected-note@-2 {{Access of 'TenElements' with a tainted offset that may be negative or too large}}
}

void arrayOverflow(void) {
  TenElements[12] = 5;
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at index 12, while it holds only 10 'int' elements}}
}

void flippedOverflow(void) {
  12[TenElements] = 5;
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at index 12, while it holds only 10 'int' elements}}
}

int *afterTheEndPtr(void) {
  // This is an unusual but standard-compliant way of writing (TenElements + 10).
  return &TenElements[10]; // no-warning
}

int useAfterTheEndPtr(void) {
  // ... but dereferencing the after-the-end pointer is still invalid.
  return *afterTheEndPtr();
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at index 10, while it holds only 10 'int' elements}}
}

int *afterAfterTheEndPtr(void) {
  // This is UB, it's invalid to form an after-after-the-end pointer.
  return &TenElements[11];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at index 11, while it holds only 10 'int' elements}}
}

int *potentialAfterTheEndPtr(int idx) {
  if (idx < 10) { /* ...do something... */ }
  // expected-note@-1 {{Assuming 'idx' is >= 10}}
  // expected-note@-2 {{Taking false branch}}
  return &TenElements[idx];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
  // NOTE: On the idx >= 10 branch the normal "optimistic" behavior would've
  // been continuing with the assumption that idx == 10 and the return value is
  // a legitimate after-the-end pointer. The checker deviates from this by
  // reporting an error because this situation is very suspicious and far from
  // the idiomatic `&TenElements[size]` expressions. If the report is FP, the
  // developer can easily silence it by writing TenElements+idx instead of
  // &TenElements[idx].
}

int overflowOrUnderflow(int arg) {
  // expected-note@+2 {{Assuming 'arg' is < 0}}
  // expected-note@+1 {{Taking false branch}}
  if (arg >= 0)
    return 0;

  return TenElements[arg - 1];
  // expected-warning@-1 {{Out of bound access to memory around 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at a negative or overflowing index, while it holds only 10 'int' elements}}
}

char TwoElements[2] = {11, 22};
char overflowOrUnderflowConcrete(int arg) {
  // expected-note@#cond {{Assuming 'arg' is < 3}}
  // expected-note@#cond {{Left side of '||' is false}}
  // expected-note@#cond {{Assuming 'arg' is not equal to 0}}
  // expected-note@#cond {{Left side of '||' is false}}
  // expected-note@#cond {{Assuming 'arg' is not equal to 1}}
  // expected-note@#cond {{Taking false branch}}
  if (arg >= 3 || arg == 0 || arg == 1) // #cond
    return 0;

  return TwoElements[arg];
  // expected-warning@-1 {{Out of bound access to memory around 'TwoElements'}}
  // expected-note@-2 {{Access of 'TwoElements' at a negative or overflowing index, while it holds only 2 'char' elements}}
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

struct item {
  int a, b;
} itemArray[20] = {0};

int arrayOfStructs(void) {
  return itemArray[35].a;
  // expected-warning@-1 {{Out of bound access to memory after the end of 'itemArray'}}
  // expected-note@-2 {{Access of 'itemArray' at index 35, while it holds only 20 'struct item' elements}}
}

int arrayOfStructsArrow(void) {
  return (itemArray + 35)->b;
  // expected-warning@-1 {{Out of bound access to memory after the end of 'itemArray'}}
  // expected-note@-2 {{Access of 'itemArray' at index 35, while it holds only 20 'struct item' elements}}
}

short convertedArray(void) {
  return ((short*)TenElements)[47];
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at index 47, while it holds only 20 'short' elements}}
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
  // However, this is reported with indices/elements, because the extent
  // (of the string that consists of 'a', 'b', 'c' and '\0') happens to be a
  // multiple of 4 bytes (= sizeof(int)).
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

int *custom_calloc(size_t a, size_t b) {
  size_t res;

  return __builtin_mul_overflow(a, b, &res) ? 0 : malloc(res);
}

int *mallocRegionOverflow(void) {
  int *mem = (int*)custom_calloc(10, sizeof(int));

  mem[20] = 10;
  // expected-warning@-1 {{Out of bound access to memory after the end of the heap area}}
  // expected-note@-2 {{Access of the heap area at index 20, while it holds only 10 'int' elements}}
  return mem;
}

int *mallocRegionDeref(void) {
  int *mem = (int*)malloc(2*sizeof(int));

  *(mem + 3) = -2;
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

int *symbolicExtent(int arg) {
  // expected-note@+2 {{Assuming 'arg' is < 5}}
  // expected-note@+1 {{Taking false branch}}
  if (arg >= 5)
    return 0;
  int *mem = (int*)malloc(arg);

  // TODO: without the following reference to 'arg', the analyzer would discard
  // the range information about (the symbolic value of) 'arg'. This is
  // incorrect because while the variable itself is inaccessible, it becomes
  // the symbolic extent of 'mem', so we still want to reason about its
  // potential values.
  (void)arg;

  mem[8] = -2;
  // expected-warning@-1 {{Out of bound access to memory after the end of the heap area}}
  // expected-note@-2 {{Access of 'int' element in the heap area at index 8}}
  return mem;
}

int *symbolicExtentDiscardedRangeInfo(int arg) {
  // This is a copy of the case 'symbolicExtent' without the '(void)arg' hack.
  // TODO: if the analyzer can detect the out-of-bounds access within this
  // testcase, then remove this and the `(void)arg` hack from `symbolicExtent`.
  if (arg >= 5)
    return 0;
  int *mem = (int*)malloc(arg);
  mem[8] = -2;
  return mem;
}

void symbolicIndex(int arg) {
  // expected-note@+2 {{Assuming 'arg' is >= 12}}
  // expected-note@+1 {{Taking true branch}}
  if (arg >= 12)
    TenElements[arg] = -2;
  // expected-warning@-1 {{Out of bound access to memory after the end of 'TenElements'}}
  // expected-note@-2 {{Access of 'TenElements' at an overflowing index, while it holds only 10 'int' elements}}
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
  // but apparently the analyzer isn't smart enough to deduce this.

  // Keep constraints alive. (Without this, the overeager garbage collection of
  // constraints would _also_ prevent the intended behavior in this testcase.)
  (void)x;

  return mem;
}
