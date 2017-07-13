// RUN: %clang_analyze_cc1 -analyzer-checker=unix.StdCLibraryFunctions,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -triple i686-unknown-linux -analyzer-checker=unix.StdCLibraryFunctions,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -triple x86_64-unknown-linux -analyzer-checker=unix.StdCLibraryFunctions,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -triple armv7-a15-linux -analyzer-checker=unix.StdCLibraryFunctions,debug.ExprInspection -verify %s
// RUN: %clang_analyze_cc1 -triple thumbv7-a15-linux -analyzer-checker=unix.StdCLibraryFunctions,debug.ExprInspection -verify %s

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

size_t fread(void *, size_t, size_t, FILE *);
size_t fwrite(const void *restrict, size_t, size_t, FILE *restrict);
void test_fread_fwrite(FILE *fp, int *buf) {
  size_t x = fwrite(buf, sizeof(int), 10, fp);
  clang_analyzer_eval(x <= 10); // expected-warning{{TRUE}}
  size_t y = fread(buf, sizeof(int), 10, fp);
  clang_analyzer_eval(y <= 10); // expected-warning{{TRUE}}
  size_t z = fwrite(buf, sizeof(int), y, fp);
  // FIXME: should be TRUE once symbol-symbol constraint support is improved.
  clang_analyzer_eval(z <= y); // expected-warning{{UNKNOWN}}
}

ssize_t getline(char **, size_t *, FILE *);
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
void test_getchar() {
  int x = getchar();
  if (x == EOF)
    return;
  clang_analyzer_eval(x < 0); // expected-warning{{FALSE}}
  clang_analyzer_eval(x < 256); // expected-warning{{TRUE}}
}

int isalpha(int);
void test_isalpha() {
  clang_analyzer_eval(isalpha(']')); // expected-warning{{FALSE}}
  clang_analyzer_eval(isalpha('Q')); // expected-warning{{TRUE}}
  clang_analyzer_eval(isalpha(128)); // expected-warning{{UNKNOWN}}
}

int isalnum(int);
void test_alnum() {
  clang_analyzer_eval(isalnum('1')); // expected-warning{{TRUE}}
  clang_analyzer_eval(isalnum(')')); // expected-warning{{FALSE}}
}

int isblank(int);
void test_isblank() {
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

void test_call_by_pointer() {
  typedef int (*func)(int);
  func f = isascii;
  clang_analyzer_eval(f('A')); // expected-warning{{TRUE}}
  f = ispunct;
  clang_analyzer_eval(f('A')); // expected-warning{{FALSE}}
}
