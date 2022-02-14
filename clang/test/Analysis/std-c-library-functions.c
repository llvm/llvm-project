// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple x86_64-unknown-linux \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple armv7-a15-linux \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple thumbv7-a15-linux \
// RUN:   -verify

// RUN: %clang_analyze_cc1 %s \
// RUN:   -analyzer-checker=core \
// RUN:   -analyzer-checker=apiModeling.StdCLibraryFunctions \
// RUN:   -analyzer-config apiModeling.StdCLibraryFunctions:DisplayLoadedSummaries=true \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false \
// RUN:   -triple i686-unknown-linux 2>&1 | FileCheck %s

//      CHECK: Loaded summary for: int isalnum(int)
// CHECK-NEXT: Loaded summary for: int isalpha(int)
// CHECK-NEXT: Loaded summary for: int isascii(int)
// CHECK-NEXT: Loaded summary for: int isblank(int)
// CHECK-NEXT: Loaded summary for: int isdigit(int)
// CHECK-NEXT: Loaded summary for: int isgraph(int)
// CHECK-NEXT: Loaded summary for: int islower(int)
// CHECK-NEXT: Loaded summary for: int isprint(int)
// CHECK-NEXT: Loaded summary for: int ispunct(int)
// CHECK-NEXT: Loaded summary for: int isspace(int)
// CHECK-NEXT: Loaded summary for: int isupper(int)
// CHECK-NEXT: Loaded summary for: int isxdigit(int)
// CHECK-NEXT: Loaded summary for: int getc(FILE *)
// CHECK-NEXT: Loaded summary for: int fgetc(FILE *)
// CHECK-NEXT: Loaded summary for: int getchar()
// CHECK-NEXT: Loaded summary for: unsigned int fread(void *restrict, size_t, size_t, FILE *restrict)
// CHECK-NEXT: Loaded summary for: unsigned int fwrite(const void *restrict, size_t, size_t, FILE *restrict)
// CHECK-NEXT: Loaded summary for: ssize_t read(int, void *, size_t)
// CHECK-NEXT: Loaded summary for: ssize_t write(int, const void *, size_t)
// CHECK-NEXT: Loaded summary for: ssize_t getline(char **restrict, size_t *restrict, FILE *restrict)
// CHECK-NEXT: Loaded summary for: ssize_t getdelim(char **restrict, size_t *restrict, int, FILE *restrict)


void clang_analyzer_eval(int);

int glob;

typedef struct FILE FILE;
#define EOF -1

int getc(FILE *);
void test_getc(FILE *fp) {
  int x;
  while ((x = getc(fp)) != EOF) {
    clang_analyzer_eval(x > 255); // expected-warning{{FALSE}}
    clang_analyzer_eval(x >= 0); // expected-warning{{TRUE}}
  }
}

int fgetc(FILE *);
void test_fgets(FILE *fp) {
  clang_analyzer_eval(fgetc(fp) < 256); // expected-warning{{TRUE}}
  clang_analyzer_eval(fgetc(fp) >= 0); // expected-warning{{UNKNOWN}}
}


typedef typeof(sizeof(int)) size_t;
typedef signed long ssize_t;
ssize_t read(int, void *, size_t);
ssize_t write(int, const void *, size_t);
void test_read_write(int fd, char *buf) {
  glob = 1;
  ssize_t x = write(fd, buf, 10);
  clang_analyzer_eval(glob); // expected-warning{{UNKNOWN}}
  if (x >= 0) {
    clang_analyzer_eval(x <= 10); // expected-warning{{TRUE}}
    ssize_t y = read(fd, &glob, sizeof(glob));
    if (y >= 0) {
      clang_analyzer_eval(y <= sizeof(glob)); // expected-warning{{TRUE}}
    } else {
      // -1 overflows on promotion!
      clang_analyzer_eval(y <= sizeof(glob)); // expected-warning{{FALSE}}
    }
  } else {
    clang_analyzer_eval(x == -1); // expected-warning{{TRUE}}
  }
}

size_t fread(void *restrict, size_t, size_t, FILE *restrict);
size_t fwrite(const void *restrict, size_t, size_t, FILE *restrict);
void test_fread_fwrite(FILE *fp, int *buf) {

  size_t x = fwrite(buf, sizeof(int), 10, fp);
  clang_analyzer_eval(x <= 10); // expected-warning{{TRUE}}

  size_t y = fread(buf, sizeof(int), 10, fp);
  clang_analyzer_eval(y <= 10); // expected-warning{{TRUE}}

  size_t z = fwrite(buf, sizeof(int), y, fp);
  clang_analyzer_eval(z <= y); // expected-warning{{TRUE}}
}

void test_fread_uninitialized(void) {
  void *ptr;
  size_t sz;
  size_t nmem;
  FILE *fp;
  (void)fread(ptr, sz, nmem, fp); // expected-warning {{1st function call argument is an uninitialized value}}
}

ssize_t getline(char **restrict, size_t *restrict, FILE *restrict);
ssize_t getdelim(char **restrict, size_t *restrict, int, FILE *restrict);
void test_getline(FILE *fp) {
  char *line = 0;
  size_t n = 0;
  ssize_t len;
  while ((len = getline(&line, &n, fp)) != -1) {
    clang_analyzer_eval(len == 0); // expected-warning{{FALSE}}
  }
}

int isascii(int);
void test_isascii(int x) {
  clang_analyzer_eval(isascii(123)); // expected-warning{{TRUE}}
  clang_analyzer_eval(isascii(-1)); // expected-warning{{FALSE}}
  if (isascii(x)) {
    clang_analyzer_eval(x < 128); // expected-warning{{TRUE}}
    clang_analyzer_eval(x >= 0); // expected-warning{{TRUE}}
  } else {
    if (x > 42)
      clang_analyzer_eval(x >= 128); // expected-warning{{TRUE}}
    else
      clang_analyzer_eval(x < 0); // expected-warning{{TRUE}}
  }
  glob = 1;
  isascii('a');
  clang_analyzer_eval(glob); // expected-warning{{TRUE}}
}

int islower(int);
void test_islower(int x) {
  clang_analyzer_eval(islower('x')); // expected-warning{{TRUE}}
  clang_analyzer_eval(islower('X')); // expected-warning{{FALSE}}
  if (islower(x))
    clang_analyzer_eval(x < 'a'); // expected-warning{{FALSE}}
}

int getchar(void);
void test_getchar(void) {
  int x = getchar();
  if (x == EOF)
    return;
  clang_analyzer_eval(x < 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(x < 256); // expected-warning{{TRUE}}
}

int isalpha(int);
void test_isalpha(void) {
  clang_analyzer_eval(isalpha(']')); // expected-warning{{FALSE}}
  clang_analyzer_eval(isalpha('Q')); // expected-warning{{TRUE}}
  clang_analyzer_eval(isalpha(128)); // expected-warning{{UNKNOWN}}
}

int isalnum(int);
void test_alnum(void) {
  clang_analyzer_eval(isalnum('1')); // expected-warning{{TRUE}}
  clang_analyzer_eval(isalnum(')')); // expected-warning{{FALSE}}
}

int isblank(int);
void test_isblank(void) {
  clang_analyzer_eval(isblank('\t')); // expected-warning{{TRUE}}
  clang_analyzer_eval(isblank(' ')); // expected-warning{{TRUE}}
  clang_analyzer_eval(isblank('\n')); // expected-warning{{FALSE}}
}

int ispunct(int);
void test_ispunct(int x) {
  clang_analyzer_eval(ispunct(' ')); // expected-warning{{FALSE}}
  clang_analyzer_eval(ispunct(-1)); // expected-warning{{FALSE}}
  clang_analyzer_eval(ispunct('#')); // expected-warning{{TRUE}}
  clang_analyzer_eval(ispunct('_')); // expected-warning{{TRUE}}
  if (ispunct(x))
    clang_analyzer_eval(x < 127); // expected-warning{{TRUE}}
}

int isupper(int);
void test_isupper(int x) {
  if (isupper(x))
    clang_analyzer_eval(x < 'A'); // expected-warning{{FALSE}}
}

int isgraph(int);
int isprint(int);
void test_isgraph_isprint(int x) {
  char y = x;
  if (isgraph(y))
    clang_analyzer_eval(isprint(x)); // expected-warning{{TRUE}}
}

int isdigit(int);
void test_mixed_branches(int x) {
  if (isdigit(x)) {
    clang_analyzer_eval(isgraph(x)); // expected-warning{{TRUE}}
    clang_analyzer_eval(isblank(x)); // expected-warning{{FALSE}}
  } else if (isascii(x)) {
    // isalnum() bifurcates here.
    clang_analyzer_eval(isalnum(x)); // expected-warning{{TRUE}} // expected-warning{{FALSE}}
    clang_analyzer_eval(isprint(x)); // expected-warning{{TRUE}} // expected-warning{{FALSE}}
  }
}

int isspace(int);
void test_isspace(int x) {
  if (!isascii(x))
    return;
  char y = x;
  if (y == ' ')
    clang_analyzer_eval(isspace(x)); // expected-warning{{TRUE}}
}

int isxdigit(int);
void test_isxdigit(int x) {
  if (isxdigit(x) && isupper(x)) {
    clang_analyzer_eval(x >= 'A'); // expected-warning{{TRUE}}
    clang_analyzer_eval(x <= 'F'); // expected-warning{{TRUE}}
  }
}

void test_call_by_pointer(void) {
  typedef int (*func)(int);
  func f = isascii;
  clang_analyzer_eval(f('A')); // expected-warning{{TRUE}}
  f = ispunct;
  clang_analyzer_eval(f('A')); // expected-warning{{FALSE}}
}

char *getenv(const char *name);
void test_getenv(void) {
  // getenv() bifurcates here.
  clang_analyzer_eval(getenv("FOO") == 0);
  // expected-warning@-1 {{TRUE}}
  // expected-warning@-2 {{FALSE}}
}
