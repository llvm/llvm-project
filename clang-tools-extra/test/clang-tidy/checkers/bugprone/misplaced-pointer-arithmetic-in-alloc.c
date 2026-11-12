// RUN: %check_clang_tidy %s bugprone-misplaced-pointer-arithmetic-in-alloc %t

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *alloca(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);

void bad_malloc(int n) {
  char *p = (char *)malloc(n) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: arithmetic operation is applied to the result of malloc() instead of its size-like argument
  // CHECK-FIXES: char *p = (char *)malloc(n + 10);

  p = (char *)malloc(n) - 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of malloc() instead of its size-like argument
  // CHECK-FIXES: p = (char *)malloc(n - 10);

  p = (char *)malloc(n) + n;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of malloc() instead of its size-like argument
  // CHECK-FIXES: p = (char *)malloc(n + n);

  p = (char *)malloc(n) - (n + 10);
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of malloc() instead of its size-like argument
  // CHECK-FIXES: p = (char *)malloc(n - (n + 10));

  p = (char *)malloc(n) - n + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: arithmetic operation is applied to the result of malloc() instead of its size-like argument
  // CHECK-FIXES: p = (char *)malloc(n - n) + 10;
  // FIXME: should be p = (char *)malloc(n - n + 10);
}

void bad_alloca(int n) {
  char *p = (char *)alloca(n) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: arithmetic operation is applied to the result of alloca() instead of its size-like argument
  // CHECK-FIXES: char *p = (char *)alloca(n + 10);
}

void bad_realloc(char *s, int n) {
  char *p = (char *)realloc(s, n) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: arithmetic operation is applied to the result of realloc() instead of its size-like argument
  // CHECK-FIXES: char *p = (char *)realloc(s, n + 10);
}

void bad_calloc(int n, int m) {
  char *p = (char *)calloc(m, n) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: arithmetic operation is applied to the result of calloc() instead of its size-like argument
  // CHECK-FIXES: char *p = (char *)calloc(m, n + 10);
}

void (*(*const alloc_ptr)(size_t)) = malloc;

void bad_indirect_alloc(int n) {
  char *p = (char *)alloc_ptr(n) + 10;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: arithmetic operation is applied to the result of alloc_ptr() instead of its size-like argument
  // CHECK-FIXES: char *p = (char *)alloc_ptr(n + 10);
}
