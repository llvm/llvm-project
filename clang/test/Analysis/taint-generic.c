// RUN: %clang_analyze_cc1 -Wno-format-security -Wno-pointer-to-int-cast \
// RUN:   -Wno-incompatible-library-redeclaration -verify %s \
// RUN:   -analyzer-checker=alpha.security.taint \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.security.ArrayBoundV2 \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config \
// RUN:     alpha.security.taint.TaintPropagation:Config=%S/Inputs/taint-generic-config.yaml

// RUN: %clang_analyze_cc1 -Wno-format-security -Wno-pointer-to-int-cast \
// RUN:   -Wno-incompatible-library-redeclaration -verify %s \
// RUN:   -DFILE_IS_STRUCT \
// RUN:   -analyzer-checker=alpha.security.taint \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=alpha.security.ArrayBoundV2 \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config \
// RUN:     alpha.security.taint.TaintPropagation:Config=%S/Inputs/taint-generic-config.yaml

// RUN: not %clang_analyze_cc1 -Wno-pointer-to-int-cast \
// RUN:   -Wno-incompatible-library-redeclaration -verify %s \
// RUN:   -analyzer-checker=alpha.security.taint \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config \
// RUN:     alpha.security.taint.TaintPropagation:Config=justguessit \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-FILE

// CHECK-INVALID-FILE: (frontend): invalid input for checker option
// CHECK-INVALID-FILE-SAME:        'alpha.security.taint.TaintPropagation:Config',
// CHECK-INVALID-FILE-SAME:        that expects a valid filename instead of
// CHECK-INVALID-FILE-SAME:        'justguessit'

// RUN: not %clang_analyze_cc1 -Wno-incompatible-library-redeclaration \
// RUN:   -verify %s \
// RUN:   -analyzer-checker=alpha.security.taint \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config \
// RUN:     alpha.security.taint.TaintPropagation:Config=%S/Inputs/taint-generic-config-ill-formed.yaml \
// RUN:   2>&1 | FileCheck -DMSG=%errc_EINVAL %s -check-prefix=CHECK-ILL-FORMED

// CHECK-ILL-FORMED: (frontend): invalid input for checker option
// CHECK-ILL-FORMED-SAME:        'alpha.security.taint.TaintPropagation:Config',
// CHECK-ILL-FORMED-SAME:        that expects a valid yaml file: [[MSG]]

// RUN: not %clang_analyze_cc1 -Wno-incompatible-library-redeclaration \
// RUN:   -verify %s \
// RUN:   -analyzer-checker=alpha.security.taint \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config \
// RUN:     alpha.security.taint.TaintPropagation:Config=%S/Inputs/taint-generic-config-invalid-arg.yaml \
// RUN:   2>&1 | FileCheck %s -check-prefix=CHECK-INVALID-ARG

// CHECK-INVALID-ARG: (frontend): invalid input for checker option
// CHECK-INVALID-ARG-SAME:        'alpha.security.taint.TaintPropagation:Config',
// CHECK-INVALID-ARG-SAME:        that expects an argument number for propagation
// CHECK-INVALID-ARG-SAME:        rules greater or equal to -1

typedef long long rsize_t;
void clang_analyzer_isTainted_char(char);
void clang_analyzer_isTainted_charp(char*);
void clang_analyzer_isTainted_int(int);

int scanf(const char *restrict format, ...);
char *gets(char *str);
char *gets_s(char *str, rsize_t n);
int getchar(void);

typedef struct _FILE FILE;
#ifdef FILE_IS_STRUCT
extern struct _FILE *stdin;
#else
extern FILE *stdin;
#endif

#define bool _Bool
#define NULL (void*)0

char *getenv(const char *name);

FILE *fopen(const char *name, const char *mode);

int fscanf(FILE *restrict stream, const char *restrict format, ...);
int sprintf(char *str, const char *format, ...);
void setproctitle(const char *fmt, ...);
void setproctitle_init(int argc, char *argv[], char *envp[]);
typedef __typeof(sizeof(int)) size_t;
typedef signed long long ssize_t;

// Define string functions. Use builtin for some of them. They all default to
// the processing in the taint checker.
#define strcpy(dest, src) \
  ((__builtin_object_size(dest, 0) != -1ULL) \
   ? __builtin___strcpy_chk (dest, src, __builtin_object_size(dest, 1)) \
   : __inline_strcpy_chk(dest, src))

static char *__inline_strcpy_chk (char *dest, const char *src) {
  return __builtin___strcpy_chk(dest, src, __builtin_object_size(dest, 1));
}
char *stpcpy(char *restrict s1, const char *restrict s2);
char *strncpy( char * destination, const char * source, size_t num );
char *strndup(const char *s, size_t n);
char *strncat(char *restrict s1, const char *restrict s2, size_t n);

void *malloc(size_t);
void *calloc(size_t nmemb, size_t size);
void bcopy(void *s1, void *s2, size_t n);

typedef size_t socklen_t;

struct sockaddr {
  unsigned short sa_family;
  char sa_data[14];
};

#define BUFSIZE 10

int Buffer[BUFSIZE];
void bufferScanfDirect(void)
{
  int n;
  scanf("%d", &n);
  Buffer[n] = 1; // expected-warning {{Out of bound memory access }}
}

void bufferScanfArithmetic1(int x) {
  int n;
  scanf("%d", &n);
  int m = (n - 3);
  Buffer[m] = 1; // expected-warning {{Out of bound memory access }}
}

void bufferScanfArithmetic2(int x) {
  int n;
  scanf("%d", &n);
  int m = 100 - (n + 3) * x;
  Buffer[m] = 1; // expected-warning {{Out of bound memory access }}
}

void bufferScanfAssignment(int x) {
  int n;
  scanf("%d", &n);
  int m;
  if (x > 0) {
    m = n;
    Buffer[m] = 1; // expected-warning {{Out of bound memory access }}
  }
}

void scanfArg(void) {
  int t = 0;
  scanf("%d", t); // expected-warning {{format specifies type 'int *' but the argument has type 'int'}}
}

void bufferGetchar(int x) {
  int m = getchar();
  Buffer[m] = 1;  //expected-warning {{Out of bound memory access (index is tainted)}}
}

extern const unsigned short int **__ctype_b_loc (void);
enum { _ISdigit = 2048 };
# define isdigit(c) ((*__ctype_b_loc ())[(int) (c)] & (unsigned short int) _ISdigit)

int isdigitImplFalsePositive(void) {
  // If this code no longer produces a bug report, then consider removing the
  // special case that disables buffer overflow reports coming from the isXXXXX
  // macros in ctypes.h.
  int c = getchar();
  return ((*__ctype_b_loc ())[(int) (c)] & (unsigned short int) _ISdigit);
  //expected-warning@-1 {{Out of bound memory access (index is tainted)}}
}

int isdigitSuppressed(void) {
  // Same code as above, but reports are suppressed based on macro name:
  int c = getchar();
  return isdigit(c); //no-warning
}

// Some later tests use isdigit as a function, so we need to undef it:
#undef isdigit

void testUncontrolledFormatString(char **p) {
  char s[80];
  fscanf(stdin, "%s", s);
  char buf[128];
  sprintf(buf,s); // expected-warning {{Uncontrolled Format String}}
  setproctitle(s, 3); // expected-warning {{Uncontrolled Format String}}

  // Test taint propagation through strcpy and family.
  char scpy[80];
  strcpy(scpy, s);
  sprintf(buf,scpy); // expected-warning {{Uncontrolled Format String}}

  stpcpy(*(++p), s); // this generates __inline.
  setproctitle(*(p), 3); // expected-warning {{Uncontrolled Format String}}

  char spcpy[80];
  stpcpy(spcpy, s);
  setproctitle(spcpy, 3); // expected-warning {{Uncontrolled Format String}}

  char *spcpyret;
  spcpyret = stpcpy(spcpy, s);
  setproctitle(spcpyret, 3); // expected-warning {{Uncontrolled Format String}}

  char sncpy[80];
  strncpy(sncpy, s, 20);
  setproctitle(sncpy, 3); // expected-warning {{Uncontrolled Format String}}

  char *dup;
  dup = strndup(s, 20);
  setproctitle(dup, 3); // expected-warning {{Uncontrolled Format String}}

}

int system(const char *command);
void testTaintSystemCall(void) {
  char buffer[156];
  char addr[128];
  scanf("%s", addr);
  system(addr); // expected-warning {{Untrusted data is passed to a system call}}

  // Test that spintf transfers taint.
  sprintf(buffer, "/bin/mail %s < /tmp/email", addr);
  system(buffer); // expected-warning {{Untrusted data is passed to a system call}}
}

void testTaintSystemCall2(void) {
  // Test that snpintf transfers taint.
  char buffern[156];
  char addr[128];
  scanf("%s", addr);
  __builtin_snprintf(buffern, 10, "/bin/mail %s < /tmp/email", addr);
  // expected-warning@-1 {{'snprintf' will always be truncated; specified size is 10, but format string expands to at least 24}}
  system(buffern); // expected-warning {{Untrusted data is passed to a system call}}
}

void testTaintSystemCall3(void) {
  char buffern2[156];
  int numt;
  char addr[128];
  scanf("%s %d", addr, &numt);
  __builtin_snprintf(buffern2, numt, "/bin/mail %s < /tmp/email", "abcd");
  system(buffern2); // expected-warning {{Untrusted data is passed to a system call}}
}

void testGets(void) {
  char str[50];
  gets(str);
  system(str); // expected-warning {{Untrusted data is passed to a system call}}
}

void testGets_s(void) {
  char str[50];
  gets_s(str, 49);
  system(str); // expected-warning {{Untrusted data is passed to a system call}}
}

void testTaintedBufferSize(void) {
  size_t ts;
  scanf("%zd", &ts);

  int *buf1 = (int*)malloc(ts*sizeof(int)); // expected-warning {{Untrusted data is used to specify the buffer size}}
  char *dst = (char*)calloc(ts, sizeof(char)); //expected-warning {{Untrusted data is used to specify the buffer size}}
  bcopy(buf1, dst, ts); // expected-warning {{Untrusted data is used to specify the buffer size}}
  __builtin_memcpy(dst, buf1, (ts + 4)*sizeof(char)); // expected-warning {{Untrusted data is used to specify the buffer size}}

  // If both buffers are trusted, do not issue a warning.
  char *dst2 = (char*)malloc(ts*sizeof(char)); // expected-warning {{Untrusted data is used to specify the buffer size}}
  strncat(dst2, dst, ts); // no-warning
}

#define AF_UNIX   1   /* local to host (pipes) */
#define AF_INET   2   /* internetwork: UDP, TCP, etc. */
#define AF_LOCAL  AF_UNIX   /* backward compatibility */
#define SOCK_STREAM 1
int socket(int, int, int);
size_t read(int, void *, size_t);
int  execl(const char *, const char *, ...);

void testSocket(void) {
  int sock;
  char buffer[100];

  sock = socket(AF_INET, SOCK_STREAM, 0);
  read(sock, buffer, 100);
  execl(buffer, "filename", 0); // expected-warning {{Untrusted data is passed to a system call}}

  sock = socket(AF_LOCAL, SOCK_STREAM, 0);
  read(sock, buffer, 100);
  execl(buffer, "filename", 0); // no-warning

  sock = socket(AF_INET, SOCK_STREAM, 0);
  // References to both buffer and &buffer as an argument should taint the argument
  read(sock, &buffer, 100);
  execl(buffer, "filename", 0); // expected-warning {{Untrusted data is passed to a system call}}
}

void testStruct(void) {
  struct {
    char buf[16];
    int length;
  } tainted;

  char buffer[16];
  int sock;

  sock = socket(AF_INET, SOCK_STREAM, 0);
  read(sock, &tainted, sizeof(tainted));
  __builtin_memcpy(buffer, tainted.buf, tainted.length); // expected-warning {{Untrusted data is used to specify the buffer size}}
}

void testStructArray(void) {
  struct {
    int length;
  } tainted[4];

  char dstbuf[16], srcbuf[16];
  int sock;

  sock = socket(AF_INET, SOCK_STREAM, 0);
  __builtin_memset(srcbuf, 0, sizeof(srcbuf));

  read(sock, &tainted[0], sizeof(tainted));
  __builtin_memcpy(dstbuf, srcbuf, tainted[0].length); // expected-warning {{Untrusted data is used to specify the buffer size}}

  __builtin_memset(&tainted, 0, sizeof(tainted));
  read(sock, &tainted, sizeof(tainted));
  __builtin_memcpy(dstbuf, srcbuf, tainted[0].length); // expected-warning {{Untrusted data is used to specify the buffer size}}

  __builtin_memset(&tainted, 0, sizeof(tainted));
  // If we taint element 1, we should not raise an alert on taint for element 0 or element 2
  read(sock, &tainted[1], sizeof(tainted));
  __builtin_memcpy(dstbuf, srcbuf, tainted[0].length); // no-warning
  __builtin_memcpy(dstbuf, srcbuf, tainted[2].length); // no-warning
}

void testUnion(void) {
  union {
    int x;
    char y[4];
  } tainted;

  char buffer[4];

  int sock = socket(AF_INET, SOCK_STREAM, 0);
  read(sock, &tainted.y, sizeof(tainted.y));
  // FIXME: overlapping regions aren't detected by isTainted yet
  __builtin_memcpy(buffer, tainted.y, tainted.x);
}

int testDivByZero(void) {
  int x;
  scanf("%d", &x);
  return 5/x; // expected-warning {{Division by a tainted value, possibly zero}}
}

// Zero-sized VLAs.
void testTaintedVLASize(void) {
  int x;
  scanf("%d", &x);
  int vla[x]; // expected-warning{{Declared variable-length array (VLA) has tainted size}}
}

int testTaintedAllocaMem() {
  char x;
  void * p;
  scanf("%c", &x);
  p = __builtin_alloca(1);
  __builtin_memcpy(p, &x, 1);
  return 5 / *(char*)p; // expected-warning {{Division by a tainted value, possibly zero}}
}

int testTaintedMallocMem() {
  char x;
  void * p;
  scanf("%c", &x);
  p = malloc(1);
  __builtin_memcpy(p, &x, 1);
  return 5 / *(char*)p; // expected-warning {{Division by a tainted value, possibly zero}}
}


// This computation used to take a very long time.
#define longcmp(a,b,c) { \
  a -= c;  a ^= c;  c += b; b -= a;  b ^= (a<<6) | (a >> (32-b));  a += c; c -= b;  c ^= b;  b += a; \
  a -= c;  a ^= c;  c += b; b -= a;  b ^= a;  a += c; c -= b;  c ^= b;  b += a; }

unsigned radar11369570_hanging(const unsigned char *arr, int l) {
  unsigned a, b, c;
  a = b = c = 0x9899e3 + l;
  while (l >= 6) {
    unsigned t;
    scanf("%d", &t);
    a += b;
    a ^= a;
    a += (arr[3] + ((unsigned) arr[2] << 8) + ((unsigned) arr[1] << 16) + ((unsigned) arr[0] << 24));
    longcmp(a, t, c);
    l -= 12;
  }
  return 5/a; // expected-warning {{Division by a tainted value, possibly zero}}
}

// Check that we do not assert of the following code.
int SymSymExprWithDiffTypes(void* p) {
  int i;
  scanf("%d", &i);
  int j = (i % (int)(long)p);
  return 5/j; // expected-warning {{Division by a tainted value, possibly zero}}
}


void constraintManagerShouldTreatAsOpaque(int rhs) {
  int i;
  scanf("%d", &i);
  // This comparison used to hit an assertion in the constraint manager,
  // which didn't handle NonLoc sym-sym comparisons.
  if (i < rhs)
    return;
  if (i < rhs)
    *(volatile int *) 0; // no-warning
}

int testSprintf_is_not_a_source(char *buf, char *msg) {
  int x = sprintf(buf, "%s", msg); // no-warning
  return 1 / x;                    // no-warning: 'sprintf' is not a taint source
}

int testSprintf_propagates_taint(char *buf, char *msg) {
  scanf("%s", msg);
  int x = sprintf(buf, "%s", msg); // propagate taint!
  return 1 / x;                    // expected-warning {{Division by a tainted value, possibly zero}}
}

int scanf_s(const char *format, ...);
int testScanf_s_(int *out) {
  scanf_s("%d", out);
  return 1 / *out; // expected-warning {{Division by a tainted value, possibly zero}}
}

#define _IO_FILE FILE
int _IO_getc(_IO_FILE *__fp);
int testUnderscoreIO_getc(_IO_FILE *fp) {
  char c = _IO_getc(fp);
  return 1 / c; // expected-warning {{Division by a tainted value, possibly zero}}
}

char *getcwd(char *buf, size_t size);
int testGetcwd(char *buf, size_t size) {
  char *c = getcwd(buf, size);
  return system(c); // expected-warning {{Untrusted data is passed to a system call}}
}

char *getwd(char *buf);
int testGetwd(char *buf) {
  char *c = getwd(buf);
  return system(c); // expected-warning {{Untrusted data is passed to a system call}}
}

ssize_t readlink(const char *path, char *buf, size_t bufsiz);
int testReadlink(char *path, char *buf, size_t bufsiz) {
  ssize_t s = readlink(path, buf, bufsiz);
  system(buf); // expected-warning {{Untrusted data is passed to a system call}}
  // readlink never returns 0
  return 1 / (s + 1); // expected-warning {{Division by a tainted value, possibly zero}}
}

ssize_t readlinkat(int dirfd, const char *pathname, char *buf, size_t bufsiz);
int testReadlinkat(int dirfd, char *path, char *buf, size_t bufsiz) {
  ssize_t s = readlinkat(dirfd, path, buf, bufsiz);
  system(buf);        // expected-warning {{Untrusted data is passed to a system call}}
  (void)(1 / dirfd);  // arg 0 is not tainted
  system(path);       // arg 1 is not tainted
  (void)(1 / bufsiz); // arg 3 is not tainted
  // readlinkat never returns 0
  return 1 / (s + 1); // expected-warning {{Division by a tainted value, possibly zero}}
}

char *get_current_dir_name(void);
int testGet_current_dir_name() {
  char *d = get_current_dir_name();
  return system(d); // expected-warning {{Untrusted data is passed to a system call}}
}

int gethostname(char *name, size_t len);
int testGethostname(char *name, size_t len) {
  gethostname(name, len);
  return system(name); // expected-warning {{Untrusted data is passed to a system call}}
}

int getnameinfo(const struct sockaddr *restrict addr, socklen_t addrlen,
                char *restrict host, socklen_t hostlen,
                char *restrict serv, socklen_t servlen, int flags);
int testGetnameinfo(const struct sockaddr *restrict addr, socklen_t addrlen,
                    char *restrict host, socklen_t hostlen,
                    char *restrict serv, socklen_t servlen, int flags) {
  getnameinfo(addr, addrlen, host, hostlen, serv, servlen, flags);

  system(host);        // expected-warning {{Untrusted data is passed to a system call}}
  return system(serv); // expected-warning {{Untrusted data is passed to a system call}}
}

int getseuserbyname(const char *linuxuser, char **selinuxuser, char **level);
int testGetseuserbyname(const char *linuxuser, char **selinuxuser, char **level) {
  getseuserbyname(linuxuser, selinuxuser, level);
  system(selinuxuser[0]);  // expected-warning {{Untrusted data is passed to a system call}}
  return system(level[0]); // expected-warning {{Untrusted data is passed to a system call}}
}

typedef int gid_t;
int getgroups(int size, gid_t list[]);
int testGetgroups(int size, gid_t list[], bool flag) {
  int result = getgroups(size, list);
  if (flag)
    return 1 / list[0]; // expected-warning {{Division by a tainted value, possibly zero}}

  return 1 / (result + 1); // expected-warning {{Division by a tainted value, possibly zero}}
}

char *getlogin(void);
int testGetlogin() {
  char *n = getlogin();
  return system(n); // expected-warning {{Untrusted data is passed to a system call}}
}

int getlogin_r(char *buf, size_t bufsize);
int testGetlogin_r(char *buf, size_t bufsize) {
  getlogin_r(buf, bufsize);
  return system(buf); // expected-warning {{Untrusted data is passed to a system call}}
}

int fscanf_s(FILE *stream, const char *format, ...);
void testFscanf_s(const char *fname, int *d) {
  FILE *f = fopen(fname, "r");
  fscanf_s(f, "%d", d);
  clang_analyzer_isTainted_int(*d); // expected-warning {{YES}}
}

int fread(void *buffer, size_t size, size_t count, FILE *stream);
void testFread(const char *fname, int *buffer, size_t size, size_t count) {
  FILE *f = fopen(fname, "r");
  size_t read = fread(buffer, size, count, f);

  clang_analyzer_isTainted_int(*buffer); // expected-warning {{YES}}
  clang_analyzer_isTainted_int(read); // expected-warning {{YES}}
}

ssize_t recv(int sockfd, void *buf, size_t len, int flags);
int accept(int fd, struct sockaddr *addr, socklen_t *addrlen);
int bind(int fd, const struct sockaddr *addr, socklen_t addrlen);
int listen(int fd, int backlog);

void testRecv(int *buf, size_t len, int flags) {
  int fd;
  scanf("%d", &fd); // fake a tainted a file descriptor

  size_t read = recv(fd, buf, len, flags);
  clang_analyzer_isTainted_int(*buf); // expected-warning {{YES}}
  clang_analyzer_isTainted_int(read); // expected-warning {{YES}}
}

ssize_t recvfrom(int sockfd, void *restrict buf, size_t len, int flags,
                 struct sockaddr *restrict src_addr,
                 socklen_t *restrict addrlen);
void testRecvfrom(int *restrict buf, size_t len, int flags,
                 struct sockaddr *restrict src_addr,
                 socklen_t *restrict addrlen) {
  int fd;
  scanf("%d", &fd); // fake a tainted a file descriptor

  size_t read = recvfrom(fd, buf, len, flags, src_addr, addrlen);
  clang_analyzer_isTainted_int(*buf); // expected-warning {{YES}}
  clang_analyzer_isTainted_int(read); // expected-warning {{YES}}
}

char *ttyname(int fd);
void testTtyname() {
  int fd;
  scanf("%d", &fd); // fake a tainted a file descriptor

  char *name = ttyname(fd);
  clang_analyzer_isTainted_charp(name); // expected-warning {{YES}}
}

int ttyname_r(int fd, char *buf, size_t buflen);
void testTtyname_r(char *buf, size_t buflen) {
  int fd;
  scanf("%d", &fd); // fake a tainted a file descriptor

  int result = ttyname_r(fd, buf, buflen);
  clang_analyzer_isTainted_char(*buf); // expected-warning {{YES}}
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

char *dirname(char *path);
void testDirname() {
  char buf[10];
  scanf("%9s", buf);

  char *name = dirname(buf);
  clang_analyzer_isTainted_charp(name); // expected-warning {{YES}}
}

char *basename(char *path);
void testBasename() {
  char buf[10];
  scanf("%9s", buf);

  char *name = basename(buf);
  clang_analyzer_isTainted_charp(name); // expected-warning {{YES}}
}

int fnmatch(const char *pattern, const char *string, int flags);
void testFnmatch(const char *pattern, int flags) {
  char string[10];
  scanf("%9s", string);

  int result = fnmatch(pattern, string, flags);
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

void *memchr(const void *s, int c, size_t n);
void testMemchr(int c, size_t n) {
  char buf[10];
  scanf("%9s", buf);

  char *result = memchr(buf, c, n);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

void *memrchr(const void *s, int c, size_t n);
void testMemrchr(int c, size_t n) {
  char buf[10];
  scanf("%9s", buf);

  char *result = memrchr(buf, c, n);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

void *rawmemchr(const void *s, int c);
void testRawmemchr(int c) {
  char buf[10];
  scanf("%9s", buf);

  char *result = rawmemchr(buf, c);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

typedef char wchar_t;
int mbtowc(wchar_t *pwc, const char *s, size_t n);
void testMbtowc(wchar_t *pwc, size_t n) {
  char buf[10];
  scanf("%9s", buf);

  int result = mbtowc(pwc, buf, n);
  clang_analyzer_isTainted_char(*pwc); // expected-warning {{YES}}
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

int wctomb(char *s, wchar_t wc);
void testWctomb(char *buf) {
  wchar_t wc;
  scanf("%c", &wc);

  int result = wctomb(buf, wc);
  clang_analyzer_isTainted_char(*buf); // expected-warning {{YES}}
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

int wcwidth(wchar_t c);
void testWcwidth() {
  wchar_t wc;
  scanf("%c", &wc);

  int width = wcwidth(wc);
  clang_analyzer_isTainted_int(width); // expected-warning {{YES}}
}

int memcmp(const void *s1, const void *s2, size_t n);
void testMemcmpWithLHSTainted(size_t n, char *rhs) {
  char lhs[10];
  scanf("%9s", lhs);

  int cmp_result = memcmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void testMemcmpWithRHSTainted(size_t n, char *lhs) {
  char rhs[10];
  scanf("%9s", rhs);

  int cmp_result = memcmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void *memcpy(void *restrict dest, const void *restrict src, size_t n);
void testMemcpy(char *dst, size_t n) {
  char src[10];
  scanf("%9s", src);

  char *result = memcpy(dst, src, n);

  clang_analyzer_isTainted_char(*dst); // expected-warning {{YES}}
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

void *memmove(void *dest, const void *src, size_t n);
void testMemmove(char *dst, size_t n) {
  char src[10];
  scanf("%9s", src);

  char *result = memmove(dst, src, n);

  clang_analyzer_isTainted_char(*dst); // expected-warning {{YES}}
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

void *memmem(const void *haystack, size_t haystacklen, const void *needle, size_t needlelen);
void testMemmem(const void *needle, size_t needlelen) {
  char haystack[10];
  scanf("%9s", haystack);

  char *result = memmem(haystack, 9, needle, needlelen);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *strstr(const char *haystack, const char *needle);
void testStrstr(const char *needle) {
  char haystack[10];
  scanf("%9s", haystack);

  char *result = strstr(haystack, needle);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *strcasestr(const char *haystack, const char *needle);
void testStrcasestr(const char *needle) {
  char haystack[10];
  scanf("%9s", haystack);

  char *result = strcasestr(haystack, needle);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *strchrnul(const char *s, int c);
void testStrchrnul() {
  char s[10];
  scanf("%9s", s);

  char *result = strchrnul(s, 9);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *index(const char *s, int c);
void testIndex() {
  char s[10];
  scanf("%9s", s);

  char *result = index(s, 9);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *rindex(const char *s, int c);
void testRindex() {
  char s[10];
  scanf("%9s", s);

  char *result = rindex(s, 9);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

int strcmp(const char *s1, const char *s2);
void testStrcmpWithLHSTainted(char *rhs) {
  char lhs[10];
  scanf("%9s", lhs);

  int cmp_result = strcmp(lhs, rhs);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void testStrcmpWithRHSTainted(char *lhs) {
  char rhs[10];
  scanf("%9s", rhs);

  int cmp_result = strcmp(lhs, rhs);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}
int strcasecmp(const char *s1, const char *s2);
void testStrcasecmpWithLHSTainted(char *rhs) {
  char lhs[10];
  scanf("%9s", lhs);

  int cmp_result = strcasecmp(lhs, rhs);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void testStrcasecmpWithRHSTainted(char *lhs) {
  char rhs[10];
  scanf("%9s", rhs);

  int cmp_result = strcasecmp(lhs, rhs);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}
int strncmp(const char *s1, const char *s2, size_t n);
void testStrncmpWithLHSTainted(char *rhs, size_t n) {
  char lhs[10];
  scanf("%9s", lhs);

  int cmp_result = strncmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void testStrncmpWithRHSTainted(char *lhs, size_t n) {
  char rhs[10];
  scanf("%9s", rhs);

  int cmp_result = strncmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void testStrncmpWithNTainted(char *lhs, char *rhs) {
  int n;
  scanf("%d", &n);

  int cmp_result = strncmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

int strncasecmp(const char *s1, const char *s2, size_t n);
void testStrncasecmpWithLHSTainted(char *rhs, size_t n) {
  char lhs[10];
  scanf("%9s", lhs);

  int cmp_result = strncmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void testStrncasecmpWithRHSTainted(char *lhs, size_t n) {
  char rhs[10];
  scanf("%9s", rhs);

  int cmp_result = strncmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

void testStrncasecmpWithNTainted(char *lhs, char *rhs) {
  int n;
  scanf("%d", &n);

  int cmp_result = strncmp(lhs, rhs, n);
  clang_analyzer_isTainted_int(cmp_result); // expected-warning {{YES}}
}

size_t strspn(const char *s, const char *accept);
void testStrspnFirstArgTainted(const char *accept) {
  char s[10];
  scanf("%9s", s);

  size_t result = strspn(s, accept);
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

void testStrspnSecondArgTainted(const char *s) {
  char accept[10];
  scanf("%9s", accept);

  size_t result = strspn(s, accept);
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

size_t strcspn(const char *s, const char *reject);
void testStrcspnFirstArgTainted(const char *reject) {
  char s[10];
  scanf("%9s", s);

  size_t result = strcspn(s, reject);
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

void testStrcspnSecondArgTainted(const char *s) {
  char reject[10];
  scanf("%9s", reject);

  size_t result = strcspn(s, reject);
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

char *strpbrk(const char *s, const char *accept);
void testStrpbrk(const char *accept) {
  char s[10];
  scanf("%9s", s);

  char *result = strpbrk(s, accept);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *strndup(const char *s, size_t n);
void testStrndup(size_t n) {
  char s[10];
  scanf("%9s", s);

  char *result = strndup(s, n);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *strdupa(const char *s);
void testStrdupa() {
  char s[10];
  scanf("%9s", s);

  char *result = strdupa(s);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

char *strndupa(const char *s, size_t n);
void testStrndupa(size_t n) {
  char s[10];
  scanf("%9s", s);

  char *result = strndupa(s, n);
  clang_analyzer_isTainted_charp(result); // expected-warning {{YES}}
}

size_t strlen(const char *s);
void testStrlen() {
  char s[10];
  scanf("%9s", s);

  size_t result = strlen(s);
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

size_t strnlen(const char *s, size_t maxlen);
void testStrnlen(size_t maxlen) {
  char s[10];
  scanf("%9s", s);

  size_t result = strnlen(s, maxlen);
  clang_analyzer_isTainted_int(result); // expected-warning {{YES}}
}

long strtol(const char *restrict nptr, char **restrict endptr, int base);
long long strtoll(const char *restrict nptr, char **restrict endptr, int base);
unsigned long int strtoul(const char *nptr, char **endptr, int base);
unsigned long long int strtoull(const char *nptr, char **endptr, int base);
void testStrtolVariants(char **restrict endptr, int base) {
  char s[10];
  scanf("%9s", s);

  long result_l = strtol(s, endptr, base);
  clang_analyzer_isTainted_int(result_l); // expected-warning {{YES}}

  long long result_ll = strtoll(s, endptr, base);
  clang_analyzer_isTainted_int(result_ll); // expected-warning {{YES}}

  unsigned long result_ul = strtoul(s, endptr, base);
  clang_analyzer_isTainted_int(result_ul); // expected-warning {{YES}}

  unsigned long long result_ull = strtoull(s, endptr, base);
  clang_analyzer_isTainted_int(result_ull); // expected-warning {{YES}}
}

int isalnum(int c);
int isalpha(int c);
int isascii(int c);
int isblank(int c);
int iscntrl(int c);
int isdigit(int c);
int isgraph(int c);
int islower(int c);
int isprint(int c);
int ispunct(int c);
int isspace(int c);
int isupper(int c);
int isxdigit(int c);

void testIsFunctions() {
  char c;
  scanf("%c", &c);

  int alnum = isalnum(c);
  clang_analyzer_isTainted_int(alnum); // expected-warning {{YES}}

  int alpha = isalpha(c);
  clang_analyzer_isTainted_int(alpha); // expected-warning {{YES}}

  int ascii = isascii(c);
  clang_analyzer_isTainted_int(ascii); // expected-warning {{YES}}

  int blank = isblank(c);
  clang_analyzer_isTainted_int(blank); // expected-warning {{YES}}

  int cntrl = iscntrl(c);
  clang_analyzer_isTainted_int(cntrl); // expected-warning {{YES}}

  int digit = isdigit(c);
  clang_analyzer_isTainted_int(digit); // expected-warning {{YES}}

  int graph = isgraph(c);
  clang_analyzer_isTainted_int(graph); // expected-warning {{YES}}

  int lower = islower(c);
  clang_analyzer_isTainted_int(lower); // expected-warning {{YES}}

  int print = isprint(c);
  clang_analyzer_isTainted_int(print); // expected-warning {{YES}}

  int punct = ispunct(c);
  clang_analyzer_isTainted_int(punct); // expected-warning {{YES}}

  int space = isspace(c);
  clang_analyzer_isTainted_int(space); // expected-warning {{YES}}

  int upper = isupper(c);
  clang_analyzer_isTainted_int(upper); // expected-warning {{YES}}

  int xdigit = isxdigit(c);
  clang_analyzer_isTainted_int(xdigit); // expected-warning {{YES}}
}

void qsort(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *));
void qsort_r(void *base, size_t nmemb, size_t size, int (*compar)(const void *, const void *, void *), void *arg);
void testQsort() {
  int data[1];
  scanf("%d", data);

  qsort(data, sizeof(data), sizeof(data[0]), NULL);
  clang_analyzer_isTainted_int(data[0]); // expected-warning {{YES}}
  qsort_r(data, sizeof(data), sizeof(data[0]), NULL, NULL);
  clang_analyzer_isTainted_int(data[0]); // expected-warning {{YES}}
}

// Test configuration
int mySource1(void);
void mySource2(int*);
void myScanf(const char*, ...);
int myPropagator(int, int*);
int mySnprintf(char*, size_t, const char*, ...);
bool isOutOfRange(const int*); // const filter function
void sanitizeCmd(char*); // non-const filter function
void mySink(int, int, int);

void testConfigurationSources1(void) {
  int x = mySource1();
  Buffer[x] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationSources2(void) {
  int x;
  mySource2(&x);
  Buffer[x] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationSources3(void) {
  int x, y;
  myScanf("%d %d", &x, &y);
  Buffer[y] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationPropagation(void) {
  int x = mySource1();
  int y;
  myPropagator(x, &y);
  Buffer[y] = 1; // expected-warning {{Out of bound memory access }}
}

void testConfigurationFilter(void) {
  int x = mySource1();
  if (isOutOfRange(&x)) // the filter function
    return;
  Buffer[x] = 1; // no-warning
}

void testConfigurationFilterNonConst(void) {
  char buffer[1000];
  myScanf("%s", buffer); // makes buffer tainted
  system(buffer); // expected-warning {{Untrusted data is passed to a system call}}
}

void testConfigurationFilterNonConst2(void) {
  char buffer[1000];
  myScanf("%s", buffer); // makes buffer tainted
  sanitizeCmd(buffer); // removes taintedness
  system(buffer); // no-warning
}

void testConfigurationSinks(void) {
  int x = mySource1();
  mySink(x, 1, 2);
  // expected-warning@-1 {{Untrusted data is passed to a user-defined sink}}
  mySink(1, x, 2); // no-warning
  mySink(1, 2, x);
  // expected-warning@-1 {{Untrusted data is passed to a user-defined sink}}
}

void testUnknownFunction(void (*foo)(void)) {
  foo(); // no-crash
}

void testProctitleFalseNegative(void) {
  char flag[80];
  fscanf(stdin, "%79s", flag);
  char *argv[] = {"myapp", flag};
  // FIXME: We should have a warning below: Untrusted data passed to sink.
  setproctitle_init(1, argv, 0);
}

void testProctitle2(char *real_argv[]) {
  char *app = getenv("APP_NAME");
  if (!app)
    return;
  char *argv[] = {app, "--foobar"};
  setproctitle_init(1, argv, 0);         // expected-warning {{Untrusted data is passed to a user-defined sink}}
  setproctitle_init(1, real_argv, argv); // expected-warning {{Untrusted data is passed to a user-defined sink}}
}

void testAcceptPropagates() {
  int listenSocket = socket(2, 1, 6);
  clang_analyzer_isTainted_int(listenSocket); // expected-warning {{YES}}
  int acceptSocket = accept(listenSocket, 0, 0);
  clang_analyzer_isTainted_int(acceptSocket); // expected-warning {{YES}}
}
