// RUN: %check_clang_tidy %s cert-err33-c %t

typedef __SIZE_TYPE__ size_t;
void *aligned_alloc(size_t alignment, size_t size);
void test_aligned_alloc(void) {
  aligned_alloc(2, 10);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
}

long strtol(const char *restrict nptr, char **restrict endptr, int base);
void test_strtol(void) {
  strtol("123", 0, 10);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
}

typedef char wchar_t;
int wscanf_s(const wchar_t *restrict format, ...);
void test_wscanf_s(void) {
  int Val;
  wscanf_s("%i", &Val);
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: the value returned by this function should not be disregarded; neglecting it may lead to errors
  // CHECK-MESSAGES: [[@LINE-2]]:3: note: cast the expression to void to silence this warning
}
