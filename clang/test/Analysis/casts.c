// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -verify %s
// RUN: %clang_cc1 -triple i386-apple-darwin9 -analyze -analyzer-checker=core,alpha.core -analyzer-store=region -verify %s
// expected-no-diagnostics

// Test if the 'storage' region gets properly initialized after it is cast to
// 'struct sockaddr *'. 

typedef unsigned char __uint8_t;
typedef unsigned int __uint32_t;
typedef __uint32_t __darwin_socklen_t;
typedef __uint8_t sa_family_t;
typedef __darwin_socklen_t socklen_t;
struct sockaddr { sa_family_t sa_family; };
struct sockaddr_storage {};

void getsockname();

void f(int sock) {
  struct sockaddr_storage storage;
  struct sockaddr* sockaddr = (struct sockaddr*)&storage;
  socklen_t addrlen = sizeof(storage);
  getsockname(sock, sockaddr, &addrlen);
  switch (sockaddr->sa_family) { // no-warning
  default:
    ;
  }
}

struct s {
  struct s *value;
};

void f1(struct s **pval) {
  int *tbool = ((void*)0);
  struct s *t = *pval;
  pval = &(t->value);
  tbool = (int *)pval; // use the cast-to type 'int *' to create element region.
  char c = (unsigned char) *tbool; // Should use cast-to type to create symbol.
  if (*tbool == -1) // here load the element region with the correct type 'int'
    (void)3;
}

void f2(const char *str) {
 unsigned char ch, cl, *p;

 p = (unsigned char *)str;
 ch = *p++; // use cast-to type 'unsigned char' to create element region.
 cl = *p++;
 if(!cl)
    cl = 'a';
}

// Test cast VariableSizeArray to pointer does not crash.
void *memcpy(void *, void const *, unsigned long);
typedef unsigned char Byte;
void doit(char *data, int len) {
    if (len) {
        Byte buf[len];
        memcpy(buf, data, len);
    }
}

// PR 6013 and 6035 - Test that a cast of a pointer to long and then to int does not crash SValuator.
void pr6013_6035_test(void *p) {
  unsigned int foo;
  foo = ((long)(p));
  (void) foo;
}

// PR12511 and radar://11215362 - Test that we support SymCastExpr, which represents symbolic int to float cast.
char ttt(int intSeconds) {
  double seconds = intSeconds;
  if (seconds)
    return 0;
  return 0;
}
