// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,unix.DynamicMemoryModeling \
// RUN:   -analyzer-config unix.DynamicMemoryModeling:ModelAllocationFailure=true
// RUN: %clang_analyze_cc1 -verify=nofailurebranch %s \
// RUN:   -analyzer-checker=core,unix.DynamicMemoryModeling \
// RUN:   -analyzer-config unix.DynamicMemoryModeling:ModelAllocationFailure=false

// nofailurebranch-no-diagnostics

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *alloca(size_t);
void *valloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);
void *calloc(size_t nmemb, size_t size);
char *strdup(const char *s);
struct if_nameindex { char x; };
struct if_nameindex *if_nameindex(void);
void if_freenameindex(struct if_nameindex *ptr);

void test_malloc(size_t s) {
  char *p = malloc(s);
  if (s > 0) {
    *p = 1; //expected-warning{{Dereference of null pointer}}
  } else {
    *p = 1;
  }
  free(p);
}

void test_calloc(size_t n, size_t s) {
  char *p = calloc(n, s);
  if (n > 0 && s > 0) {
    *p = 1; //expected-warning{{Dereference of null pointer}}
  } else {
    *p = 1;
  }
  free(p);
}

void test_valloc(size_t s) {
  int *p = valloc(s);
  *p = 1; //no-warning
  free(p);
}

void test_alloca(size_t s) {
  int *p = alloca(s);
  *p = 1; //no-warning
}

void test_realloc(size_t s) {
  char *p = malloc(10);
  if (!p)
    return;
  p = realloc(p, s);
  if (s > 0) {
    *p = 1; //expected-warning{{Dereference of null pointer}}
  } else {
    *p = 1;
  }
  free(p);
}

void test_strdup() {
  char *p = strdup("abcd");
  *p = 1; //expected-warning{{Dereference of null pointer}}
  free(p);
}

void test_ifnameindex() {
  struct if_nameindex *p = if_nameindex();
  p->x = 1; //expected-warning{{dereference of a null pointer}}
  if_freenameindex(p);
}
