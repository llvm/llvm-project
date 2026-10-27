// Built-in list of C library and POSIX parameters that must not be null.
// A nullable argument there warns like one passed to a _Nonnull parameter, in
// any function the analysis covers. Parameters documented to accept null
// (free, realloc, fflush(NULL), strtol's endptr, ...) are not on the list.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -verify=on %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fnullability-default=nonnull -std=c11 -verify=on %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -fno-nullability-libc-nonnull-params -std=c11 -verify=off %s

// The declarations sit in a system header, as real libc ones do, so the
// nonnull default does not apply to them.
# 1 "libc.h" 1 3
typedef __SIZE_TYPE__ size_t;
typedef struct FILE FILE;
FILE *fopen(const char *, const char *);
int fclose(FILE *);
int fflush(FILE *);
int fputs(const char *, FILE *);
char *fgets(char *, int, FILE *);
size_t fread(void *, size_t, size_t, FILE *);
int fprintf(FILE *, const char *, ...);
int snprintf(char *, size_t, const char *, ...);
size_t strlen(const char *);
int strcmp(const char *, const char *);
char *strrchr(const char *, int);
char *strcpy(char *, const char *);
char *strtok(char *, const char *);
void *memcpy(void *, const void *, size_t);
long strtol(const char *, char **, int);
int atoi(const char *);
char *getenv(const char *);
void free(void *);
void *realloc(void *, size_t);
typedef long time_t;
time_t time(time_t *);
int setvbuf(FILE *, char *, int, size_t);
struct DIR;
struct DIR *opendir(const char *);
int closedir(struct DIR *);
# 41 "nullability-safety-libc-nonnull-params.c" 2

char *_Nullable maybe_str(void);
char buf[16];

// The libc nullable-return list feeds the nonnull-parameter list: the redis
// fclose(fopen(...)) and cJSON strcmp(strrchr(...)) shapes.
void close_unchecked(const char *_Nonnull path) {
  FILE *fp = fopen(path, "r");
  fclose(fp); // on-warning {{passing nullable pointer to nonnull parameter}} on-note {{}}
}
int compare_unchecked(const char *_Nonnull s) {
  return strcmp(strrchr(s, '/'), "x"); // on-warning {{passing nullable pointer to nonnull parameter}} on-note {{}}
}
size_t len_env(int *_Nullable trigger) {
  return strlen(getenv("HOME")); // on-warning {{passing nullable pointer to nonnull parameter}} on-note {{}}
}
void several(char *_Nullable s, FILE *_Nullable f) {
  fputs(s, f); // on-warning 2 {{passing nullable pointer to nonnull parameter}} on-note 2 {{}}
}
void copy(char *_Nullable dst) {
  memcpy(dst, "x", 1); // on-warning {{passing nullable pointer to nonnull parameter}} on-note {{}}
}
int parse(void) {
  return atoi(maybe_str()); // on-warning {{passing nullable pointer to nonnull parameter}} on-note {{}}
}

// Checked or provably nonnull arguments are fine.
void close_checked(const char *_Nonnull path) {
  FILE *fp = fopen(path, "r");
  if (!fp)
    return;
  fgets(buf, sizeof buf, fp);
  fread(buf, 1, sizeof buf, fp);
  fprintf(fp, "%s", path);
  fclose(fp);
}
// A call that survives narrows its argument, so only the first use warns.
size_t narrowed_after_call(char *_Nullable s) {
  size_t n = strlen(s); // on-warning {{passing nullable pointer to nonnull parameter}} on-note {{}}
  return n + strlen(s);
}

// Parameters that accept null are not on the list.
void accepts_null(char *_Nullable p, FILE *_Nullable f, char **_Nullable end) {
  free(p);
  p = realloc(p, 4);
  fflush(f);
  fflush(0);
  time(0);
  strtol("1", end, 10);
  strtok(p, ",");
  snprintf(p, 0, "%d", 1);
  FILE *g = fopen("x", "r");
  if (g)
    setvbuf(g, p, 0, 0);
}

// Unannotated pointers stay unspecified in the default mode, and are nonnull
// under the nonnull default: neither warns.
size_t unannotated(int *_Nullable trigger, char *s, struct DIR *d) {
  closedir(d);
  return strlen(s);
}

// opendir reads its path.
void dir(const char *_Nullable path) {
  struct DIR *d = opendir(path); // on-warning {{passing nullable pointer to nonnull parameter}} on-note {{}}
  if (d)
    closedir(d);
}

// off-no-diagnostics
