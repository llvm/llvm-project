// RUN: %clang_analyze_cc1  -analyzer-checker=alpha.security.taint,core,alpha.security.ArrayBoundV2 -Wno-format-security -verify %s

int scanf(const char *restrict format, ...);
int getchar(void);

typedef struct _FILE FILE;
extern FILE *stdin;
int fscanf(FILE *restrict stream, const char *restrict format, ...);
int sprintf(char *str, const char *format, ...);
void setproctitle(const char *fmt, ...);
typedef __typeof(sizeof(int)) size_t;

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

void scanfArg() {
  int t = 0;
  scanf("%d", t); // expected-warning {{format specifies type 'int *' but the argument has type 'int'}}
}

void bufferGetchar(int x) {
  int m = getchar();
  Buffer[m] = 1;  //expected-warning {{Out of bound memory access (index is tainted)}}
}

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
void testTaintSystemCall() {
  char buffer[156];
  char addr[128];
  scanf("%s", addr);
  system(addr); // expected-warning {{Untrusted data is passed to a system call}}

  // Test that spintf transfers taint.
  sprintf(buffer, "/bin/mail %s < /tmp/email", addr);
  system(buffer); // expected-warning {{Untrusted data is passed to a system call}}
}

void testTaintSystemCall2() {
  // Test that snpintf transfers taint.
  char buffern[156];
  char addr[128];
  scanf("%s", addr);
  __builtin_snprintf(buffern, 10, "/bin/mail %s < /tmp/email", addr);
  system(buffern); // expected-warning {{Untrusted data is passed to a system call}}
}

void testTaintSystemCall3() {
  char buffern2[156];
  int numt;
  char addr[128];
  scanf("%s %d", addr, &numt);
  __builtin_snprintf(buffern2, numt, "/bin/mail %s < /tmp/email", "abcd");
  system(buffern2); // expected-warning {{Untrusted data is passed to a system call}}
}

void testTaintedBufferSize() {
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

void testSocket() {
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

void testStruct() {
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

void testStructArray() {
  struct {
    char buf[16];
    struct {
      int length;
    } st[1];
  } tainted;

  char buffer[16];
  int sock;

  sock = socket(AF_INET, SOCK_STREAM, 0);
  read(sock, &tainted.buf[0], sizeof(tainted.buf));
  read(sock, &tainted.st[0], sizeof(tainted.st));
  // FIXME: tainted.st[0].length should be marked tainted
  __builtin_memcpy(buffer, tainted.buf, tainted.st[0].length); // no-warning
}

int testDivByZero() {
  int x;
  scanf("%d", &x);
  return 5/x; // expected-warning {{Division by a tainted value, possibly zero}}
}

// Zero-sized VLAs.
void testTaintedVLASize() {
  int x;
  scanf("%d", &x);
  int vla[x]; // expected-warning{{Declared variable-length array (VLA) has tainted size}}
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
