// RUN: %check_clang_tidy %s misc-suspicious-string-compare %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: misc-suspicious-string-compare.WarnOnImplicitComparison, value: 1}, \
// RUN:   {key: misc-suspicious-string-compare.WarnOnLogicalNotComparison, value: 1}]}' \
// RUN: -- -std=c99

static const char A[] = "abc";

int strcmp(const char *, const char *);

int test_warning_patterns() {
  if (strcmp(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is called without explicitly comparing result [misc-suspicious-string-compare]
  // CHECK-FIXES: if (strcmp(A, "a") != 0)

  if (strcmp(A, "a") != 0 ||
      strcmp(A, "b"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is called without explicitly comparing result
  // CHECK-FIXES: strcmp(A, "b") != 0)

  if (strcmp(A, "a") == 1)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is compared to a suspicious constant

  if (strcmp(A, "a") == -1)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is compared to a suspicious constant

  if (strcmp(A, "a") < '0')
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is compared to a suspicious constant

  if (strcmp(A, "a") < 0.)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' has suspicious implicit cast

  if (!strcmp(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:8: warning: function 'strcmp' is compared using logical not operator
  // CHECK-FIXES: if (strcmp(A, "a") == 0)
}

void test_structure_patterns() {
  if (strcmp(A, "a")) {}
  // CHECK-MESSAGES: [[@LINE-1]]:7: warning: function 'strcmp' is called without explicitly comparing result
  // CHECK-FIXES: if (strcmp(A, "a") != 0) {}

  while (strcmp(A, "a")) {}
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: function 'strcmp' is called without explicitly comparing result
  // CHECK-FIXES: while (strcmp(A, "a") != 0) {}

  for (;strcmp(A, "a");) {}
  // CHECK-MESSAGES: [[@LINE-1]]:9: warning: function 'strcmp' is called without explicitly comparing result
  // CHECK-FIXES: for (;strcmp(A, "a") != 0;) {}
}

int test_valid_patterns() {
  // The following cases are valid.
  if (strcmp(A, "a") < 0) return 0;
  if (strcmp(A, "a") == 0) return 0;
  if (strcmp(A, "a") <= 0) return 0;
  if (strcmp(A, "a") == strcmp(A, "b")) return 0;
  return 1;
}

int wrapper(const char* a, const char* b) {
  return strcmp(a, b);
}

int assignment_wrapper(const char* a, const char* b) {
  int cmp = strcmp(a, b);
  return cmp;
}

int condexpr_wrapper(const char* a, const char* b) {
  return (a < b) ? strcmp(a, b) : strcmp(b, a);
}
