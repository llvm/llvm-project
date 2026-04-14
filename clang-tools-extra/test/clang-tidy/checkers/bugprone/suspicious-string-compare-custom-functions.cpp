// RUN: %check_clang_tidy %s bugprone-suspicious-string-compare %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     bugprone-suspicious-string-compare.StringCompareLikeFunctions: 'my_strcmp;my_strncmp' \
// RUN:   }}"

int my_strcmp(const char *, const char *);
int my_strncmp(const char *, const char *, unsigned long);

void TestCustomCompareFunctions() {
  if (my_strcmp("a", "b"))
    return;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'my_strcmp' is called without explicitly comparing result [bugprone-suspicious-string-compare]
  // CHECK-FIXES: if (my_strcmp("a", "b") != 0)

  if (my_strncmp("a", "b", 1))
    return;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'my_strncmp' is called without explicitly comparing result [bugprone-suspicious-string-compare]
  // CHECK-FIXES: if (my_strncmp("a", "b", 1) != 0)

  if (my_strcmp("a", "b") == 1)
    return;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'my_strcmp' is compared to a suspicious constant [bugprone-suspicious-string-compare]

  if (my_strcmp("a", "b") == 0)
    return;
}
