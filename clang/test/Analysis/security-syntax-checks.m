// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -DUSE_BUILTINS -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -DVARIANT -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify
// RUN: %clang_analyze_cc1 -triple i386-apple-darwin10 -DUSE_BUILTINS -DVARIANT -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-cloudabi -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-cloudabi -DUSE_BUILTINS -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-cloudabi -DVARIANT -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-cloudabi -DUSE_BUILTINS -DVARIANT -analyzer-checker=security.insecureAPI,security.FloatLoopCounter %s -verify

#ifdef USE_BUILTINS
# define BUILTIN(f) __builtin_ ## f
#else /* USE_BUILTINS */
# define BUILTIN(f) f
#endif /* USE_BUILTINS */

#include "Inputs/system-header-simulator-for-valist.h"
#include "Inputs/system-header-simulator-for-simple-stream.h"

typedef typeof(sizeof(int)) size_t;


// <rdar://problem/6336718> rule request: floating point used as loop 
//  condition (FLP30-C, FLP-30-CPP)
//
// For reference: https://www.securecoding.cert.org/confluence/display/seccode/FLP30-C.+Do+not+use+floating+point+variables+as+loop+counters
//
void test_float_condition() {
  for (float x = 0.1f; x <= 1.0f; x += 0.1f) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (float x = 100000001.0f; x <= 100000010.0f; x += 1.0f) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (float x = 100000001.0f; x <= 100000010.0f; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'float'}}
  for (double x = 100000001.0; x <= 100000010.0; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  for (double x = 100000001.0; ((x)) <= 100000010.0; ((x))++ ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  for (double x = 100000001.0; 100000010.0 >= x; x = x + 1.0 ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  int i = 0;
  for (double x = 100000001.0; ((x)) <= 100000010.0; ((x))++, ++i ) {} // expected-warning{{Variable 'x' with floating point type 'double'}}
  
  typedef float FooType;
  for (FooType x = 100000001.0f; x <= 100000010.0f; x++ ) {} // expected-warning{{Variable 'x' with floating point type 'FooType'}}
}

// Obsolete function bcmp
int bcmp(const void *, const void *, size_t);

int test_bcmp(void *a, void *b, size_t n) {
  return bcmp(a, b, n); // expected-warning{{The bcmp() function is obsoleted by memcmp()}}
}

// Obsolete function bcopy
void bcopy(void *, void *, size_t);

void test_bcopy(void *a, void *b, size_t n) {
  bcopy(a, b, n); // expected-warning{{The bcopy() function is obsoleted by memcpy() or memmove(}}
}

// Obsolete function bzero
void bzero(void *, size_t);

void test_bzero(void *a, size_t n) {
  bzero(a, n); // expected-warning{{The bzero() function is obsoleted by memset()}}
}

// <rdar://problem/6335715> rule request: gets() buffer overflow
// Part of recommendation: 300-BSI (buildsecurityin.us-cert.gov)
char* gets(char *buf);

void test_gets() {
  char buff[1024];
  gets(buff); // expected-warning{{Call to function 'gets' is extremely insecure as it can always result in a buffer overflow}}
}

int getpw(unsigned int uid, char *buf);

void test_getpw() {
  char buff[1024];
  getpw(2, buff); // expected-warning{{The getpw() function is dangerous as it may overflow the provided buffer. It is obsoleted by getpwuid()}}
}

// <rdar://problem/6337132> CWE-273: Failure to Check Whether Privileges Were
//  Dropped Successfully
typedef unsigned int __uint32_t;
typedef __uint32_t __darwin_uid_t;
typedef __uint32_t __darwin_gid_t;
typedef __darwin_uid_t uid_t;
typedef __darwin_gid_t gid_t;
int setuid(uid_t);
int setregid(gid_t, gid_t);
int setreuid(uid_t, uid_t);
extern void check(int);
void abort(void);

void test_setuid() 
{
  setuid(2); // expected-warning{{The return value from the call to 'setuid' is not checked.  If an error occurs in 'setuid', the following code may execute with unexpected privileges}}
  setuid(0); // expected-warning{{The return value from the call to 'setuid' is not checked.  If an error occurs in 'setuid', the following code may execute with unexpected privileges}}
  if (setuid (2) != 0)
    abort();

  // Currently the 'setuid' check is not flow-sensitive, and only looks
  // at whether the function was called in a compound statement.  This
  // will lead to false negatives, but there should be no false positives.
  int t = setuid(2);  // no-warning
  (void)setuid (2); // no-warning

  check(setuid (2)); // no-warning

  setreuid(2,2); // expected-warning{{The return value from the call to 'setreuid' is not checked.  If an error occurs in 'setreuid', the following code may execute with unexpected privileges}}
  setregid(2,2); // expected-warning{{The return value from the call to 'setregid' is not checked.  If an error occurs in 'setregid', the following code may execute with unexpected privileges}}
}

// <rdar://problem/6337100> CWE-338: Use of cryptographically weak prng
typedef  unsigned short *ushort_ptr_t;  // Test that sugar doesn't confuse the warning.
int      rand(void);
double   drand48(void);
double   erand48(unsigned short[3]);
long     jrand48(ushort_ptr_t);
void     lcong48(unsigned short[7]);
long     lrand48(void);
long     mrand48(void);
long     nrand48(unsigned short[3]);
long     random(void);
int      rand_r(unsigned *);

void test_rand()
{
  unsigned short a[7];
  unsigned b;
  
  rand();	// expected-warning{{Function 'rand' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  drand48();	// expected-warning{{Function 'drand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  erand48(a);	// expected-warning{{Function 'erand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  jrand48(a);	// expected-warning{{Function 'jrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  lcong48(a);	// expected-warning{{Function 'lcong48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  lrand48();	// expected-warning{{Function 'lrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  mrand48();	// expected-warning{{Function 'mrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  nrand48(a);	// expected-warning{{Function 'nrand48' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  rand_r(&b);	// expected-warning{{Function 'rand_r' is obsolete because it implements a poor random number generator.  Use 'arc4random' instead}}
  random();	// expected-warning{{The 'random' function produces a sequence of values that an adversary may be able to predict.  Use 'arc4random' instead}}
}

char *mktemp(char *buf);

void test_mktemp() {
  char *x = mktemp("/tmp/zxcv"); // expected-warning{{Call to function 'mktemp' is insecure as it always creates or uses insecure temporary file}}
}


//===----------------------------------------------------------------------===
// strcpy()
//===----------------------------------------------------------------------===
#ifdef VARIANT

#define __strcpy_chk BUILTIN(__strcpy_chk)
char *__strcpy_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define strcpy(a,b) __strcpy_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define strcpy BUILTIN(strcpy)
char *strcpy(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */

void test_strcpy() {
  char x[4];
  char *y;

  strcpy(x, y); //expected-warning{{Call to function 'strcpy' is insecure as it does not provide bounding of the memory buffer. Replace unbounded copy functions with analogous functions that support length arguments such as 'strlcpy'. CWE-119}}
}

void test_strcpy_2() {
  char x[4];
  strcpy(x, "abcd"); //expected-warning{{Call to function 'strcpy' is insecure as it does not provide bounding of the memory buffer. Replace unbounded copy functions with analogous functions that support length arguments such as 'strlcpy'. CWE-119}}
}

void test_strcpy_safe() {
  char x[5];
  strcpy(x, "abcd");
}

void test_strcpy_safe_2() {
  struct {char s1[100];} s;
  strcpy(s.s1, "hello");
}

//===----------------------------------------------------------------------===
// strcat()
//===----------------------------------------------------------------------===
#ifdef VARIANT

#define __strcat_chk BUILTIN(__strcat_chk)
char *__strcat_chk(char *restrict s1, const char *restrict s2, size_t destlen);

#define strcat(a,b) __strcat_chk(a,b,(size_t)-1)

#else /* VARIANT */

#define strcat BUILTIN(strcat)
char *strcat(char *restrict s1, const char *restrict s2);

#endif /* VARIANT */

void test_strcat() {
  char x[4];
  char *y;

  strcat(x, y); //expected-warning{{Call to function 'strcat' is insecure as it does not provide bounding of the memory buffer. Replace unbounded copy functions with analogous functions that support length arguments such as 'strlcat'. CWE-119}}
}

//===----------------------------------------------------------------------===
// vfork()
//===----------------------------------------------------------------------===
typedef int __int32_t;
typedef __int32_t pid_t;
pid_t vfork(void);

void test_vfork() {
  vfork(); //expected-warning{{Call to function 'vfork' is insecure as it can lead to denial of service situations in the parent process}}
}

//===----------------------------------------------------------------------===
// mkstemp()
//===----------------------------------------------------------------------===

char *mkdtemp(char *template);
int mkstemps(char *template, int suffixlen);
int mkstemp(char *template);
char *mktemp(char *template);

void test_mkstemp() {
  mkstemp("XX"); // expected-warning {{Call to 'mkstemp' should have at least 6 'X's in the format string to be secure (2 'X's seen)}}
  mkstemp("XXXXXX");
  mkstemp("XXXXXXX");
  mkstemps("XXXXXX", 0);
  mkstemps("XXXXXX", 1); // expected-warning {{5 'X's seen}}
  mkstemps("XXXXXX", 2); // expected-warning {{Call to 'mkstemps' should have at least 6 'X's in the format string to be secure (4 'X's seen, 2 characters used as a suffix)}}
  mkdtemp("XX"); // expected-warning {{2 'X's seen}}
  mkstemp("X"); // expected-warning {{Call to 'mkstemp' should have at least 6 'X's in the format string to be secure (1 'X' seen)}}
  mkdtemp("XXXXXX");
}


//===----------------------------------------------------------------------===
// deprecated or unsafe buffer handling
//===----------------------------------------------------------------------===
typedef int wchar_t;

int sprintf(char *str, const char *format, ...);
//int vsprintf (char *s, const char *format, va_list arg);
int scanf(const char *format, ...);
int wscanf(const wchar_t *format, ...);
int fscanf(FILE *stream, const char *format, ...);
int fwscanf(FILE *stream, const wchar_t *format, ...);
int vscanf(const char *format, va_list arg);
int vwscanf(const wchar_t *format, va_list arg);
int vfscanf(FILE *stream, const char *format, va_list arg);
int vfwscanf(FILE *stream, const wchar_t *format, va_list arg);
int sscanf(const char *s, const char *format, ...);
int swscanf(const wchar_t *ws, const wchar_t *format, ...);
int vsscanf(const char *s, const char *format, va_list arg);
int vswscanf(const wchar_t *ws, const wchar_t *format, va_list arg);
int swprintf(wchar_t *ws, size_t len, const wchar_t *format, ...);
int snprintf(char *s, size_t n, const char *format, ...);
int vswprintf(wchar_t *ws, size_t len, const wchar_t *format, va_list arg);
int vsnprintf(char *s, size_t n, const char *format, va_list arg);
void *memcpy(void *destination, const void *source, size_t num);
void *memmove(void *destination, const void *source, size_t num);
char *strncpy(char *destination, const char *source, size_t num);
char *strncat(char *destination, const char *source, size_t num);
void *memset(void *ptr, int value, size_t num);

void test_deprecated_or_unsafe_buffer_handling_1() {
  char buf [5];
  wchar_t wbuf [5];
  int a;
  FILE *file;
  sprintf(buf, "a"); // expected-warning{{Call to function 'sprintf' is insecure}}
  scanf("%d", &a); // expected-warning{{Call to function 'scanf' is insecure}}
  scanf("%s", buf); // expected-warning{{Call to function 'scanf' is insecure}}
  scanf("%4s", buf); // expected-warning{{Call to function 'scanf' is insecure}}
  wscanf((const wchar_t*) L"%s", buf); // expected-warning{{Call to function 'wscanf' is insecure}}
  fscanf(file, "%d", &a); // expected-warning{{Call to function 'fscanf' is insecure}}
  fscanf(file, "%s", buf); // expected-warning{{Call to function 'fscanf' is insecure}}
  fscanf(file, "%4s", buf); // expected-warning{{Call to function 'fscanf' is insecure}}
  fwscanf(file, (const wchar_t*) L"%s", wbuf); // expected-warning{{Call to function 'fwscanf' is insecure}}
  sscanf("5", "%d", &a); // expected-warning{{Call to function 'sscanf' is insecure}}
  sscanf("5", "%s", buf); // expected-warning{{Call to function 'sscanf' is insecure}}
  sscanf("5", "%4s", buf); // expected-warning{{Call to function 'sscanf' is insecure}}
  swscanf(L"5", (const wchar_t*) L"%s", wbuf); // expected-warning{{Call to function 'swscanf' is insecure}}
  swprintf(L"5", 1, (const wchar_t*) L"%s", wbuf); // expected-warning{{Call to function 'swprintf' is insecure}}
  snprintf("5", 1, "%s", buf); // expected-warning{{Call to function 'snprintf' is insecure}}
  memcpy(buf, wbuf, 1); // expected-warning{{Call to function 'memcpy' is insecure}}
  memmove(buf, wbuf, 1); // expected-warning{{Call to function 'memmove' is insecure}}
  strncpy(buf, "a", 1); // expected-warning{{Call to function 'strncpy' is insecure}}
  strncat(buf, "a", 1); // expected-warning{{Call to function 'strncat' is insecure}}
  memset(buf, 'a', 1); // expected-warning{{Call to function 'memset' is insecure}}
}

void test_deprecated_or_unsafe_buffer_handling_2(const char *format, ...) {
  char buf [5];
  FILE *file;
  va_list args;
  va_start(args, format);
  vsprintf(buf, format, args); // expected-warning{{Call to function 'vsprintf' is insecure}}
  vscanf(format, args); // expected-warning{{Call to function 'vscanf' is insecure}}
  vfscanf(file, format, args); // expected-warning{{Call to function 'vfscanf' is insecure}}
  vsscanf("a", format, args); // expected-warning{{Call to function 'vsscanf' is insecure}}
  vsnprintf("a", 1, format, args); // expected-warning{{Call to function 'vsnprintf' is insecure}}
}

void test_deprecated_or_unsafe_buffer_handling_3(const wchar_t *format, ...) {
  wchar_t wbuf [5];
  FILE *file;
  va_list args;
  va_start(args, format);
  vwscanf(format, args); // expected-warning{{Call to function 'vwscanf' is insecure}}
  vfwscanf(file, format, args); // expected-warning{{Call to function 'vfwscanf' is insecure}}
  vswscanf(L"a", format, args); // expected-warning{{Call to function 'vswscanf' is insecure}}
  vswprintf(L"a", 1, format, args); // expected-warning{{Call to function 'vswprintf' is insecure}}
}
