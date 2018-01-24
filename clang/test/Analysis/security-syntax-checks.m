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

