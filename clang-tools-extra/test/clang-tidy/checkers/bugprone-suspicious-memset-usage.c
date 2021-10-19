// RUN: %check_clang_tidy  -expect-clang-tidy-error  %s bugprone-suspicious-memset-usage %t

void *memset(void *, int, __SIZE_TYPE__);
void *memset(void *);
// CHECK-MESSAGES: :[[@LINE-1]]:7: error: conflicting types for 'memset'

void test() {
  // no crash
  memset(undefine);
  // CHECK-MESSAGES: :[[@LINE-1]]:10: error: use of undeclared identifier
}
