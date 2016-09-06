// RUN: %clang_cc1 -fsyntax-only -verify -Wno-sizeof-array-argument %s
//
extern "C" void *bzero(void *, unsigned);
extern "C" void *memset(void *, int, unsigned);
extern "C" void *memmove(void *s1, const void *s2, unsigned n);
extern "C" void *memcpy(void *s1, const void *s2, unsigned n);
extern "C" void *memcmp(void *s1, const void *s2, unsigned n);

struct S {int a, b, c, d;};
typedef S* PS;

struct Foo {};
typedef const Foo& CFooRef;
typedef const Foo CFoo;
typedef volatile Foo VFoo;
typedef const volatile Foo CVFoo;

typedef double Mat[4][4];

template <class Dest, class Source>
inline Dest bit_cast(const Source& source) {
  Dest dest;
  memcpy(&dest, &source, sizeof(dest));
  return dest;
}

// http://www.lysator.liu.se/c/c-faq/c-2.html#2-6
void f(Mat m, const Foo& const_foo, char *buffer) {
  S s;
  S* ps = &s;
  PS ps2 = &s;
  char arr[5];
  char* parr[5];
  Foo foo;
  char* heap_buffer = new char[42];

  /* Should warn */
  memset(&s, 0, sizeof(&s));  // \
      // expected-warning {{'memset' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to remove the addressof in the argument to 'sizeof' (and multiply it by the number of elements)?}}
  memset(ps, 0, sizeof(ps));  // \
      // expected-warning {{'memset' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to dereference the argument to 'sizeof' (and multiply it by the number of elements)?}}
  memset(ps2, 0, sizeof(ps2));  // \
      // expected-warning {{'memset' call operates on objects of type 'S' while the size is based on a different type 'PS' (aka 'S *')}} expected-note{{did you mean to dereference the argument to 'sizeof' (and multiply it by the number of elements)?}}
  memset(ps2, 0, sizeof(typeof(ps2)));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same pointer type}}
  memset(ps2, 0, sizeof(PS));  // \
      // expected-warning {{argument to 'sizeof' in 'memset' call is the same pointer type}}
  memset(heap_buffer, 0, sizeof(heap_buffer));  // \
      // expected-warning {{'memset' call operates on objects of type 'char' while the size is based on a different type 'char *'}} expected-note{{did you mean to provide an explicit length?}}

  bzero(&s, sizeof(&s));  // \
      // expected-warning {{'bzero' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to remove the addressof in the argument to 'sizeof' (and multiply it by the number of elements)?}}
  bzero(ps, sizeof(ps));  // \
      // expected-warning {{'bzero' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to dereference the argument to 'sizeof' (and multiply it by the number of elements)?}}
  bzero(ps2, sizeof(ps2));  // \
      // expected-warning {{'bzero' call operates on objects of type 'S' while the size is based on a different type 'PS' (aka 'S *')}} expected-note{{did you mean to dereference the argument to 'sizeof' (and multiply it by the number of elements)?}}
  bzero(ps2, sizeof(typeof(ps2)));  // \
      // expected-warning {{argument to 'sizeof' in 'bzero' call is the same pointer type}}
  bzero(ps2, sizeof(PS));  // \
      // expected-warning {{argument to 'sizeof' in 'bzero' call is the same pointer type}}
  bzero(heap_buffer, sizeof(heap_buffer));  // \
      // expected-warning {{'bzero' call operates on objects of type 'char' while the size is based on a different type 'char *'}} expected-note{{did you mean to provide an explicit length?}}

  memcpy(&s, 0, sizeof(&s));  // \
      // expected-warning {{'memcpy' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to remove the addressof in the argument to 'sizeof' (and multiply it by the number of elements)?}}
  memcpy(0, &s, sizeof(&s));  // \
      // expected-warning {{'memcpy' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to remove the addressof in the argument to 'sizeof' (and multiply it by the number of elements)?}}

  memmove(ps, 0, sizeof(ps));  // \
      // expected-warning {{'memmove' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to dereference the argument to 'sizeof' (and multiply it by the number of elements)?}}
  memcmp(ps, 0, sizeof(ps));  // \
      // expected-warning {{'memcmp' call operates on objects of type 'S' while the size is based on a different type 'S *'}} expected-note{{did you mean to dereference the argument to 'sizeof' (and multiply it by the number of elements)?}}

  /* Shouldn't warn */
  memset((void*)&s, 0, sizeof(&s));
  memset(&s, 0, sizeof(s));
  memset(&s, 0, sizeof(S));
  memset(&s, 0, sizeof(const S));
  memset(&s, 0, sizeof(volatile S));
  memset(&s, 0, sizeof(volatile const S));
  memset(&foo, 0, sizeof(CFoo));
  memset(&foo, 0, sizeof(VFoo));
  memset(&foo, 0, sizeof(CVFoo));
  memset(ps, 0, sizeof(*ps));
  memset(ps2, 0, sizeof(*ps2));
  memset(ps2, 0, sizeof(typeof(*ps2)));
  memset(arr, 0, sizeof(arr));
  memset(parr, 0, sizeof(parr));

  bzero((void*)&s, sizeof(&s));
  bzero(&s, sizeof(s));
  bzero(&s, sizeof(S));
  bzero(&s, sizeof(const S));
  bzero(&s, sizeof(volatile S));
  bzero(&s, sizeof(volatile const S));
  bzero(&foo, sizeof(CFoo));
  bzero(&foo, sizeof(VFoo));
  bzero(&foo, sizeof(CVFoo));
  bzero(ps, sizeof(*ps));
  bzero(ps2, sizeof(*ps2));
  bzero(ps2, sizeof(typeof(*ps2)));
  bzero(arr, sizeof(arr));
  bzero(parr, sizeof(parr));

  memcpy(&foo, &const_foo, sizeof(Foo));
  memcpy((void*)&s, 0, sizeof(&s));
  memcpy(0, (void*)&s, sizeof(&s));
  char *cptr;
  memcpy(&cptr, buffer, sizeof(cptr));
  memcpy((char*)&cptr, buffer, sizeof(cptr));

  CFooRef cfoo = foo;
  memcpy(&foo, &cfoo, sizeof(Foo));

  memcpy(0, &arr, sizeof(arr));
  typedef char Buff[8];
  memcpy(0, &arr, sizeof(Buff));

  unsigned char* puc;
  bit_cast<char*>(puc);

  float* pf;
  bit_cast<int*>(pf);

  int iarr[14];
  memset(&iarr[0], 0, sizeof iarr);
  memset(iarr, 0, sizeof iarr);
  bzero(&iarr[0], sizeof iarr);
  bzero(iarr, sizeof iarr);

  int* iparr[14];
  memset(&iparr[0], 0, sizeof iparr);
  memset(iparr, 0, sizeof iparr);
  bzero(&iparr[0], sizeof iparr);
  bzero(iparr, sizeof iparr);

  memset(m, 0, sizeof(Mat));
  bzero(m, sizeof(Mat));

  // Copy to raw buffer shouldn't warn either
  memcpy(&foo, &arr, sizeof(Foo));
  memcpy(&arr, &foo, sizeof(Foo));

  // Shouldn't warn, and shouldn't crash either.
  memset(({
    if (0) {}
    while (0) {}
    for (;;) {}
    &s;
  }), 0, sizeof(s));

  bzero(({
    if (0) {}
    while (0) {}
    for (;;) {}
    &s;
  }), sizeof(s));
}

namespace ns {
void memset(void* s, char c, int n);
void bzero(void* s, int n);
void f(int* i) {
  memset(i, 0, sizeof(i));
  bzero(i, sizeof(i));
}
}

extern "C" int strncmp(const char *s1, const char *s2, unsigned n);
extern "C" int strncasecmp(const char *s1, const char *s2, unsigned n);
extern "C" char *strncpy(char *det, const char *src, unsigned n);
extern "C" char *strncat(char *dst, const char *src, unsigned n);
extern "C" char *strndup(const  char *src, unsigned n);

void strcpy_and_friends() {
  const char* FOO = "<- should be an array instead";
  const char* BAR = "<- this, too";

  strncmp(FOO, BAR, sizeof(FOO)); // \
      // expected-warning {{'strncmp' call operates on objects of type 'const char' while the size is based on a different type 'const char *'}} expected-note{{did you mean to provide an explicit length?}}
  strncasecmp(FOO, BAR, sizeof(FOO));  // \
      // expected-warning {{'strncasecmp' call operates on objects of type 'const char' while the size is based on a different type 'const char *'}} expected-note{{did you mean to provide an explicit length?}}

  char buff[80];

  strncpy(buff, BAR, sizeof(BAR)); // \
      // expected-warning {{'strncpy' call operates on objects of type 'const char' while the size is based on a different type 'const char *'}} expected-note{{did you mean to provide an explicit length?}}
  strndup(FOO, sizeof(FOO)); // \
      // expected-warning {{'strndup' call operates on objects of type 'const char' while the size is based on a different type 'const char *'}} expected-note{{did you mean to provide an explicit length?}}
}
