// RUN: rm -f %t
// RUN: %clang_analyze_cc1 -fblocks -analyzer-checker=core,unix.Malloc -analyzer-output=plist -verify -o %t -analyzer-config eagerly-assume=false %s
// RUN: tail -n +11 %t | %diff_plist %S/Inputs/expected-plists/malloc-plist.c.plist -

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void free(void *);
void *realloc(void *ptr, size_t size);

void diagnosticTest(int in) {
    if (in > 5) {
        int *p = malloc(12);
        *p = 0;
        (*p)++;
    }
    in++; // expected-warning {{leak}}
}

void myArrayAllocation() {
    int **A;
    A = malloc(2*sizeof(int*));
    A[0] = 0;
}//expected-warning{{Potential leak}}

void reallocDiagnostics() {
    char * buf = malloc(100);
    char * tmp;
    tmp = (char*)realloc(buf, 0x1000000);
    if (!tmp) {
        return;// expected-warning {{leak}}
    }
    buf = tmp;
    free(buf);
}

void *wrapper() {
  void *x = malloc(100);
  // This is intentionally done to test diagnostic emission.
  if (x)
    return x;
  return 0;
}

void test_wrapper() {
  void *buf = wrapper();
  (void) buf;
}//expected-warning{{Potential leak}}

// Test what happens when the same call frees and allocated memory.
// Also tests the stack hint for parameters, when they are passed directly or via pointer.
void my_free(void *x) {
    free(x);
}
void my_malloc_and_free(void **x) {
    *x = malloc(100);
    if (*x)
      my_free(*x);
    return;
}
void *test_double_action_call() {
    void *buf;
    my_malloc_and_free(&buf);
    return buf; //expected-warning{{Use of memory after it is freed}}
}

// Test stack hint for 'reallocation failed'.
char *my_realloc(char *buf) {
    char *tmp;
    tmp = (char*)realloc(buf, 0x1000000);
    if (!tmp) {
        return tmp;
    }
    return tmp;
}
void reallocIntra() {
    char *buf = (char *)malloc(100);
    buf = my_realloc(buf);
    free(buf);//expected-warning{{Potential leak}}
}

// Test stack hint when returning a result.
static char *malloc_wrapper_ret() {
    return (char*)malloc(12);
}
void use_ret() {
    char *v;
    v = malloc_wrapper_ret();
}//expected-warning{{Potential leak}}

// Passing a block as a parameter to an inlined call for which we generate
// a stack hint message caused crashes.
// rdar://problem/21291971
void myfree_takingblock(void (^ignored)(void), int *p) {
  free(p);
}

void call_myfree_takingblock() {
  void (^some_block)(void) = ^void(void) { };

  int *p = malloc(sizeof(int));
  myfree_takingblock(some_block, p);
  *p = 3;//expected-warning{{Use of memory after it is freed}}
}

// Test that we refer to the last symbol used in the leak diagnostic.
void LeakedSymbol(int in) {
    int *m = 0;
    int *p;
    p = (int*)malloc(12);
    *p = 0;
    (*p)++;
    m = p;
    p = 0;
    (*m)++;
    in++;//expected-warning{{Potential leak}}
}

// Tests that exercise running remove dead bindings at Call exit.
static void function_with_leak1() {
    char *x = (char*)malloc(12);
} //expected-warning{{Potential leak}}
void use_function_with_leak1() {
    function_with_leak1();
    int y = 0;
}

static void function_with_leak2() {
    char *x = (char*)malloc(12);
    int m = 0; //expected-warning{{Potential leak}}
}
void use_function_with_leak2() {
    function_with_leak2();
}

static void function_with_leak3(int y) {
    char *x = (char*)malloc(12);
    if (y)
        y++;
}//expected-warning{{Potential leak}}
void use_function_with_leak3(int y) {
    function_with_leak3(y);
}

static void function_with_leak4(int y) {
    char *x = (char*)malloc(12);
    if (y)
        y++;
    else
        y--;//expected-warning{{Potential leak}}
}
void use_function_with_leak4(int y) {
    function_with_leak4(y);
}

int anotherFunction5() {
    return 5;
}
static int function_with_leak5() {
    char *x = (char*)malloc(12);
    return anotherFunction5();//expected-warning{{Potential leak}}
}
void use_function_with_leak5() {
    function_with_leak5();
}

void anotherFunction6(int m) {
    m++;
}
static void function_with_leak6() {
    char *x = (char*)malloc(12);
    anotherFunction6(3);//expected-warning{{Potential leak}}
}
void use_function_with_leak6() {
    function_with_leak6();
}

static void empty_function(){
}
void use_empty_function() {
    empty_function();
}
static char *function_with_leak7() {
    return (char*)malloc(12);
}
void use_function_with_leak7() {
    function_with_leak7();
}//expected-warning{{Potential memory leak}}

// Test that we do not print the name of a variable not visible from where
// the issue is reported.
int *my_malloc() {
  int *p = malloc(12);
  return p;
}
void testOnlyRefferToVisibleVariables() {
  my_malloc();
} // expected-warning{{Potential memory leak}}

struct PointerWrapper{
  int*p;
};
int *my_malloc_into_struct() {
  struct PointerWrapper w;
  w.p = malloc(12);
  return w.p;
}
void testMyMalloc() {
  my_malloc_into_struct();
} // expected-warning{{Potential memory leak}}
