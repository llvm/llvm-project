// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64-apple-darwin9 -verify %s

int a[10];

int f0() {
  return __builtin_object_size(&a); // expected-error {{too few arguments to function}}
}
int f1() {
  return (__builtin_object_size(&a, 0) + 
          __builtin_object_size(&a, 1) + 
          __builtin_object_size(&a, 2) + 
          __builtin_object_size(&a, 3));
}
int f2() {
  return __builtin_object_size(&a, -1); // expected-error {{argument should be a value from 0 to 3}}
}
int f3() {
  return __builtin_object_size(&a, 4); // expected-error {{argument should be a value from 0 to 3}}
}


// rdar://6252231 - cannot call vsnprintf with va_list on x86_64
void f4(const char *fmt, ...) {
 __builtin_va_list args;
 __builtin___vsnprintf_chk (0, 42, 0, 11, fmt, args); // expected-warning {{'__builtin___vsnprintf_chk' will always overflow destination buffer}}
}

// rdar://18334276
typedef __typeof__(sizeof(int)) size_t;
void * memcset(void *restrict dst, int src, size_t n);
void * memcpy(void *restrict dst, const void *restrict src, size_t n);

#define memset(dest, src, len) __builtin___memset_chk(dest, src, len, __builtin_object_size(dest, 0))
#define memcpy(dest, src, len) __builtin___memcpy_chk(dest, src, len, __builtin_object_size(dest, 0))
#define memcpy1(dest, src, len) __builtin___memcpy_chk(dest, src, len, __builtin_object_size(dest, 4))
#define NULL ((void *)0)

void f5(void)
{
  char buf[10];
  memset((void *)0x100000000ULL, 0, 0x1000);
  memcpy((char *)NULL + 0x10000, buf, 0x10);
  memcpy1((char *)NULL + 0x10000, buf, 0x10); // expected-error {{argument should be a value from 0 to 3}}
}

// rdar://18431336
void f6(void)
{
  char b[5];
  char buf[10];
  __builtin___memccpy_chk (buf, b, '\0', sizeof(b), __builtin_object_size (buf, 0));
  __builtin___memccpy_chk (b, buf, '\0', sizeof(buf), __builtin_object_size (b, 0));  // expected-warning {{'__builtin___memccpy_chk' will always overflow destination buffer}}
}

int pr28314(void) {
  struct {
    struct InvalidField a; // expected-error{{has incomplete type}} expected-note 3{{forward declaration of 'struct InvalidField'}}
    char b[0];
  } *p;

  struct {
    struct InvalidField a; // expected-error{{has incomplete type}}
    char b[1];
  } *p2;

  struct {
    struct InvalidField a; // expected-error{{has incomplete type}}
    char b[2];
  } *p3;

  int a = 0;
  a += __builtin_object_size(&p->a, 0);
  a += __builtin_object_size(p->b, 0);
  a += __builtin_object_size(p2->b, 0);
  a += __builtin_object_size(p3->b, 0);
  return a;
}

int pr31843() {
  int n = 0;

  struct { int f; } a;
  int b;
  n += __builtin_object_size(({&(b ? &a : &a)->f; pr31843;}), 0); // expected-warning{{expression result unused}}

  struct statfs { char f_mntonname[1024];};
  struct statfs *outStatFSBuf;
  n += __builtin_object_size(outStatFSBuf->f_mntonname ? "" : "", 1); // expected-warning{{address of array}}
  n += __builtin_object_size(outStatFSBuf->f_mntonname ?: "", 1);

  return n;
}

typedef struct {
  char string[512];
} NestedArrayStruct;

typedef struct {
  int x;
  NestedArrayStruct session[];
} IncompleteArrayStruct;

void rd36094951_IAS_builtin_object_size_assertion(IncompleteArrayStruct *p) {
#define rd36094951_CHECK(mode)                                                 \
  __builtin___strlcpy_chk(p->session[0].string, "ab", 2,                       \
                          __builtin_object_size(p->session[0].string, mode))
  rd36094951_CHECK(0);
  rd36094951_CHECK(1);
  rd36094951_CHECK(2);
  rd36094951_CHECK(3);
}
