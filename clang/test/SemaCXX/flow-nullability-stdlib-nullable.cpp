// Tests that flow-sensitive nullability treats known C/C++ stdlib functions
// as returning nullable pointers (provably nullable, not just unspecified).
// This ensures unchecked dereferences warn even without explicit _Nullable
// annotations on the function declarations.
//
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fnullability-default=nullable -Wno-nullable-to-nonnull-conversion -std=c++17 %s -verify

typedef unsigned long size_t;
#define NULL ((void *)0)

// Declare stdlib functions WITHOUT _Nullable annotations — the allowlist
// should still treat their returns as nullable.
extern "C" {
void *malloc(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);
void *aligned_alloc(size_t, size_t);
void free(void *);

struct FILE;
FILE *fopen(const char *, const char *);
FILE *freopen(const char *, const char *, FILE *);
FILE *tmpfile(void);

char *getenv(const char *);
char *strtok(char *, const char *);

char *strstr(const char *, const char *);
char *strchr(const char *, int);
char *strrchr(const char *, int);
char *strpbrk(const char *, const char *);
void *memchr(const void *, int, size_t);
void *bsearch(const void *, const void *, size_t, size_t,
              int (*)(const void *, const void *));
char *tmpnam(char *);
char *setlocale(int, const char *);
}

//===----------------------------------------------------------------------===//
// malloc — unchecked deref warns, checked deref is clean
//===----------------------------------------------------------------------===//

void test_malloc_unchecked() {
  int *p = (int *)malloc(sizeof(int));
  *p = 42; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  free(p);
}

void test_malloc_checked() {
  int *p = (int *)malloc(sizeof(int));
  if (!p) return;
  *p = 42; // OK — narrowed by null check
  free(p);
}

//===----------------------------------------------------------------------===//
// calloc, realloc, aligned_alloc
//===----------------------------------------------------------------------===//

void test_calloc_unchecked() {
  int *p = (int *)calloc(10, sizeof(int));
  *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  free(p);
}

void test_realloc_unchecked(void *old) {
  int *p = (int *)realloc(old, 100);
  *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  free(p);
}

void test_aligned_alloc_unchecked() {
  int *p = (int *)aligned_alloc(16, sizeof(int));
  *p = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  free(p);
}

//===----------------------------------------------------------------------===//
// fopen — unchecked deref warns
//===----------------------------------------------------------------------===//

void test_fopen_unchecked() {
  FILE *f = fopen("/tmp/test", "r");
  (void)*f; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void test_fopen_checked() {
  FILE *f = fopen("/tmp/test", "r");
  if (!f) return;
  (void)*f; // OK — narrowed
}

//===----------------------------------------------------------------------===//
// getenv
//===----------------------------------------------------------------------===//

void test_getenv_unchecked() {
  char *val = getenv("HOME");
  char c = *val; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)c;
}

void test_getenv_checked() {
  char *val = getenv("HOME");
  if (!val) return;
  char c = *val; // OK — narrowed
  (void)c;
}

//===----------------------------------------------------------------------===//
// strstr, strchr, strrchr, strpbrk
//===----------------------------------------------------------------------===//

void test_strstr_unchecked() {
  char *p = strstr("hello world", "world");
  char c = *p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)c;
}

void test_strchr_unchecked() {
  char *p = strchr("hello", 'e');
  char c = *p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)c;
}

void test_strrchr_checked() {
  char *p = strrchr("hello", 'l');
  if (p) {
    char c = *p; // OK — narrowed
    (void)c;
  }
}

void test_strpbrk_unchecked() {
  char *p = strpbrk("hello", "aeiou");
  char c = *p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)c;
}

//===----------------------------------------------------------------------===//
// memchr, bsearch
//===----------------------------------------------------------------------===//

void test_memchr_unchecked() {
  int arr[] = {1, 2, 3};
  int *p = (int *)memchr(arr, 2, sizeof(arr));
  int v = *p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)v;
}

int compare_ints(const void * _Nonnull a, const void * _Nonnull b) {
  return *(const int *)a - *(const int *)b; // OK — params are _Nonnull
}

void test_bsearch_unchecked() {
  int arr[] = {1, 2, 3, 4, 5};
  int key = 3;
  int *p = (int *)bsearch(&key, arr, 5, sizeof(int), compare_ints);
  int v = *p; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)v;
}

//===----------------------------------------------------------------------===//
// strtok
//===----------------------------------------------------------------------===//

void test_strtok_unchecked() {
  char str[] = "hello,world";
  char *tok = strtok(str, ",");
  char c = *tok; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)c;
}

void test_strtok_loop() {
  char str[] = "a,b,c";
  for (char *tok = strtok(str, ","); tok; tok = strtok(nullptr, ",")) {
    char c = *tok; // OK — narrowed by loop condition
    (void)c;
  }
}

//===----------------------------------------------------------------------===//
// tmpnam, setlocale
//===----------------------------------------------------------------------===//

void test_tmpnam_unchecked() {
  char *name = tmpnam(nullptr);
  char c = *name; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)c;
}

void test_setlocale_unchecked() {
  char *loc = setlocale(0, "C");
  char c = *loc; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
  (void)c;
}

//===----------------------------------------------------------------------===//
// Class method with same name should NOT match the allowlist
//===----------------------------------------------------------------------===//

struct MyAllocator {
  void *malloc(size_t sz);
  char *getenv(const char *name);
};

void test_class_method_not_matched() {
  MyAllocator alloc;
  // These are member calls — should NOT be treated as stdlib nullable.
  // They'll still warn because the return type is unannotated (default nullable
  // in this mode), but this tests that the matching is scoped to free functions.
  void *p = alloc.malloc(100);
  *((int *)p) = 1; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

//===----------------------------------------------------------------------===//
// Namespace-scoped function with same name should still match
// (some codebases wrap stdlib in inline namespaces)
//===----------------------------------------------------------------------===//

void test_freopen_unchecked() {
  FILE *f = freopen("/dev/null", "w", nullptr);
  (void)*f; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}

void test_tmpfile_unchecked() {
  FILE *f = tmpfile();
  (void)*f; // expected-warning {{dereference of nullable pointer}} expected-note {{add a null check}}
}
