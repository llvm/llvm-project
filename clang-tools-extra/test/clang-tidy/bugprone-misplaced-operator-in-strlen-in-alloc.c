// RUN: %check_clang_tidy %s bugprone-misplaced-operator-in-strlen-in-alloc %t

typedef __typeof(sizeof(int)) size_t;
void *malloc(size_t);
void *alloca(size_t);
void *calloc(size_t, size_t);
void *realloc(void *, size_t);

size_t strlen(const char *);
size_t strnlen(const char *, size_t);
size_t strnlen_s(const char *, size_t);

typedef unsigned wchar_t;

size_t wcslen(const wchar_t *);
size_t wcsnlen(const wchar_t *, size_t);
size_t wcsnlen_s(const wchar_t *, size_t);

void bad_malloc(char *name) {
  char *new_name = (char *)malloc(strlen(name + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: addition operator is applied to the argument of strlen
  // CHECK-FIXES: {{^  char \*new_name = \(char \*\)malloc\(}}strlen(name) + 1{{\);$}}
  new_name = (char *)malloc(strnlen(name + 1, 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: addition operator is applied to the argument of strnlen
  // CHECK-FIXES: {{^  new_name = \(char \*\)malloc\(}}strnlen(name, 10) + 1{{\);$}}
  new_name = (char *)malloc(strnlen_s(name + 1, 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: addition operator is applied to the argument of strnlen_s
  // CHECK-FIXES: {{^  new_name = \(char \*\)malloc\(}}strnlen_s(name, 10) + 1{{\);$}}
}

void bad_malloc_wide(wchar_t *name) {
  wchar_t *new_name = (wchar_t *)malloc(wcslen(name + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:34: warning: addition operator is applied to the argument of wcslen
  // CHECK-FIXES: {{^  wchar_t \*new_name = \(wchar_t \*\)malloc\(}}wcslen(name) + 1{{\);$}}
  new_name = (wchar_t *)malloc(wcsnlen(name + 1, 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: addition operator is applied to the argument of wcsnlen
  // CHECK-FIXES: {{^  new_name = \(wchar_t \*\)malloc\(}}wcsnlen(name, 10) + 1{{\);$}}
  new_name = (wchar_t *)malloc(wcsnlen_s(name + 1, 10));
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: addition operator is applied to the argument of wcsnlen_s
  // CHECK-FIXES: {{^  new_name = \(wchar_t \*\)malloc\(}}wcsnlen_s(name, 10) + 1{{\);$}}
}

void bad_alloca(char *name) {
  char *new_name = (char *)alloca(strlen(name + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: addition operator is applied to the argument of strlen
  // CHECK-FIXES: {{^  char \*new_name = \(char \*\)alloca\(}}strlen(name) + 1{{\);$}}
}

void bad_calloc(char *name) {
  char *new_names = (char *)calloc(2, strlen(name + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: addition operator is applied to the argument of strlen
  // CHECK-FIXES: {{^  char \*new_names = \(char \*\)calloc\(2, }}strlen(name) + 1{{\);$}}
}

void bad_realloc(char *old_name, char *name) {
  char *new_name = (char *)realloc(old_name, strlen(name + 1));
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: addition operator is applied to the argument of strlen
  // CHECK-FIXES: {{^  char \*new_name = \(char \*\)realloc\(old_name, }}strlen(name) + 1{{\);$}}
}

void intentional1(char *name) {
  char *new_name = (char *)malloc(strlen(name + 1) + 1);
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:28: warning: addition operator is applied to the argument of strlen
  // We have + 1 outside as well so we assume this is intentional
}

void intentional2(char *name) {
  char *new_name = (char *)malloc(strlen(name + 2));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:28: warning: addition operator is applied to the argument of strlen
  // Only give warning for + 1, not + 2
}

void intentional3(char *name) {
  char *new_name = (char *)malloc(strlen((name + 1)));
  // CHECK-MESSAGES-NOT: :[[@LINE-1]]:28: warning: addition operator is applied to the argument of strlen
  // If expression is in extra parentheses, consider it as intentional
}
