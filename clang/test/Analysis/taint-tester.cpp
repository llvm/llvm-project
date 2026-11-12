// RUN: %clang_analyze_cc1  -analyzer-checker=optin.taint,debug.TaintTest %s -verify
// expected-no-diagnostics

typedef struct _FILE FILE;
typedef __typeof(sizeof(int)) size_t;
extern FILE *stdin;
typedef long ssize_t;
ssize_t getline(char ** __restrict, size_t * __restrict, FILE * __restrict);
int printf(const char * __restrict, ...);
int snprintf(char *, size_t, const char *, ...);
void free(void *ptr);

struct GetLineTestStruct {
  ssize_t getline(char ** __restrict, size_t * __restrict, FILE * __restrict);
};

void getlineTest(void) {
  FILE *fp;
  char *line = 0;
  size_t len = 0;
  ssize_t read;
  struct GetLineTestStruct T;

  while ((read = T.getline(&line, &len, stdin)) != -1) {
    printf("%s", line); // no warning
  }
  free(line);
}

class opaque;
void testOpaqueClass(opaque *obj) {
  char buf[20];
  snprintf(buf, 20, "%p", obj); // don't crash trying to load *obj
}
