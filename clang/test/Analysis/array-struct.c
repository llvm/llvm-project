// RUN: %clang_analyze_cc1 -analyzer-checker=core,alpha.core.CastToStruct -verify %s

struct s {
  int data;
  int data_array[10];
};

typedef struct {
  int data;
} STYPE;

void g(char *p);
void g1(struct s* p);

// Array to pointer conversion. Array in the struct field.
void f(void) {
  int a[10];
  int (*p)[10];
  p = &a;
  (*p)[3] = 1;
  
  struct s d;
  struct s *q;
  q = &d;
  q->data = 3;
  d.data_array[9] = 17;
}

// StringLiteral in lvalue context and pointer to array type.
// p: ElementRegion, q: StringRegion
void f2(void) {
  char *p = "/usr/local";
  char (*q)[4];
  q = &"abc";
}

// Typedef'ed struct definition.
void f3(void) {
  STYPE s;
}

// Initialize array with InitExprList.
void f4(void) {
  int a[] = { 1, 2, 3};
  int b[3] = { 1, 2 };
  struct s c[] = {{1,{1}}};
}

// Struct variable in lvalue context.
// Assign UnknownVal to the whole struct.
void f5(void) {
  struct s data;
  g1(&data);
}

// AllocaRegion test.
void f6(void) {
  char *p;
  p = __builtin_alloca(10); 
  g(p);
  char c = *p;
  p[1] = 'a';
  // Test if RegionStore::EvalBinOp converts the alloca region to element
  // region.
  p += 2;
}

struct s2;

void g2(struct s2 *p);

// Incomplete struct pointer used as function argument.
void f7(void) {
  struct s2 *p = __builtin_alloca(10);
  g2(p);
}

// sizeof() is unsigned while -1 is signed in array index.
void f8(void) {
  int a[10];
  a[sizeof(a)/sizeof(int) - 1] = 1; // no-warning
}

// Initialization of struct array elements.
void f9(void) {
  struct s a[10];
}

// Initializing array with string literal.
void f10(void) {
  char a1[4] = "abc";
  char a3[6] = "abc";
}

// Retrieve the default value of element/field region.
void f11(void) {
  struct s a;
  g1(&a);
  if (a.data == 0) // no-warning
    a.data = 1;
}

// Convert unsigned offset to signed when creating ElementRegion from 
// SymbolicRegion.
void f12(int *list) {
  unsigned i = 0;
  list[i] = 1;
}

struct s1 {
  struct s2 {
    int d;
  } e;
};

// The binding of a.e.d should not be removed. Test recursive subregion map
// building: a->e, e->d. Only then 'a' could be added to live region roots.
void f13(double timeout) {
  struct s1 a;
  a.e.d = (int) timeout;
  if (a.e.d == 10)
    a.e.d = 4;
}

struct s3 {
  int a[2];
};

static struct s3 opt;

// Test if the embedded array is retrieved correctly.
void f14(void) {
  struct s3 my_opt = opt;
}

void bar(int*);

struct s3 gets3(void) {
  struct s3 s;
  return s;
}

void accessArrayFieldNoCrash(void) {
  bar(gets3().a);
  bar((gets3().a));
  bar(((gets3().a)));  
}

// Test if the array is correctly invalidated.
void f15(void) {
  int a[10];
  bar(a);
  if (a[1]) // no-warning
    (void)1;
}

struct s3 p[1];

// Code from postgresql.
// Current cast logic of region store mistakenly leaves the final result region
// an ElementRegion of type 'char'. Then load a nonloc::SymbolVal from it and
// assigns to 'a'. 
void f16(struct s3 *p) {
  struct s3 a = *((struct s3*) ((char*) &p[0])); // expected-warning{{Casting a non-structure type to a structure type and accessing a field can lead to memory access errors or data corruption}}
}

void inv(struct s1 *);

// Invalidate the struct field.
void f17(void) {
  struct s1 t;
  int x;
  inv(&t);
  if (t.e.d)
    x = 1;
}

void read(char*);

void f18(void) {
  char *q;
  char *p = (char *) __builtin_alloca(10);
  read(p);
  q = p;
  q++;
  if (*q) { // no-warning
  }
}


// [PR13927] offsetof replacement macro flagged as "dereference of a null pointer"
int offset_of_data_array(void)
{
  return ((char *)&(((struct s*)0)->data_array)) - ((char *)0); // no-warning
}

int testPointerArithmeticOnVoid(void *bytes) {
  int p = 0;
  if (&bytes[0] == &bytes[1])
    return 6/p; // no-warning
  return 0;
}

int testRValueArraySubscriptExpr(void *bytes) {
  int *p = (int*)&bytes[0];
  *p = 0;
  if (*(int*)&bytes[0] == 0)
    return 0;
  return 5/(*p); // no-warning
}


