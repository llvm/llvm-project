// RUN: %check_clang_tidy %s bugprone-suspicious-string-compare %t -- \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: bugprone-suspicious-string-compare.WarnOnImplicitComparison, value: 1}, \
// RUN:   {key: bugprone-suspicious-string-compare.WarnOnLogicalNotComparison, value: 1}]}' \
// RUN: --

typedef __SIZE_TYPE__ size;

struct locale_t {
  void* dummy;
} locale;

static const char A[] = "abc";
static const unsigned char U[] = "abc";
static const unsigned char V[] = "xyz";
static const wchar_t W[] = L"abc";

int strlen(const char *);

int memcmp(const void *, const void *, size);
int wmemcmp(const wchar_t *, const wchar_t *, size);
int memicmp(const void *, const void *, size);
int _memicmp(const void *, const void *, size);
int _memicmp_l(const void *, const void *, size, locale_t);

int strcmp(const char *, const char *);
int strncmp(const char *, const char *, size);
int strcasecmp(const char *, const char *);
int strncasecmp(const char *, const char *, size);
int stricmp(const char *, const char *);
int strcmpi(const char *, const char *);
int strnicmp(const char *, const char *, size);
int _stricmp(const char *, const char * );
int _strnicmp(const char *, const char *, size);
int _stricmp_l(const char *, const char *, locale_t);
int _strnicmp_l(const char *, const char *, size, locale_t);

int wcscmp(const wchar_t *, const wchar_t *);
int wcsncmp(const wchar_t *, const wchar_t *, size);
int wcscasecmp(const wchar_t *, const wchar_t *);
int wcsicmp(const wchar_t *, const wchar_t *);
int wcsnicmp(const wchar_t *, const wchar_t *, size);
int _wcsicmp(const wchar_t *, const wchar_t *);
int _wcsnicmp(const wchar_t *, const wchar_t *, size);
int _wcsicmp_l(const wchar_t *, const wchar_t *, locale_t);
int _wcsnicmp_l(const wchar_t *, const wchar_t *, size, locale_t);

int _mbscmp(const unsigned char *, const unsigned char *);
int _mbsncmp(const unsigned char *, const unsigned char *, size);
int _mbsnbcmp(const unsigned char *, const unsigned char *, size);
int _mbsnbicmp(const unsigned char *, const unsigned char *, size);
int _mbsicmp(const unsigned char *, const unsigned char *);
int _mbsnicmp(const unsigned char *, const unsigned char *, size);
int _mbscmp_l(const unsigned char *, const unsigned char *, locale_t);
int _mbsncmp_l(const unsigned char *, const unsigned char *, size, locale_t);
int _mbsicmp_l(const unsigned char *, const unsigned char *, locale_t);
int _mbsnicmp_l(const unsigned char *, const unsigned char *, size, locale_t);
int _mbsnbcmp_l(const unsigned char *, const unsigned char *, size, locale_t);
int _mbsnbicmp_l(const unsigned char *, const unsigned char *, size, locale_t);

int test_warning_patterns() {
  if (strcmp(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is called without explicitly comparing result [bugprone-suspicious-string-compare]
  // CHECK-FIXES: if (strcmp(A, "a") != 0)

  if (strcmp(A, "a") == 0 ||
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

  if (strcmp(A, "a") == true)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is compared to a suspicious constant

  if (strcmp(A, "a") < '0')
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is compared to a suspicious constant

  if (strcmp(A, "a") < 0.)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' has suspicious implicit cast
}

int test_valid_patterns() {
  // The following cases are valid.
  if (strcmp(A, "a") < 0)
    return 0;
  if (strcmp(A, "a") == 0)
    return 0;
  if (strcmp(A, "a") <= 0)
    return 0;

  if (wcscmp(W, L"a") < 0)
    return 0;
  if (wcscmp(W, L"a") == 0)
    return 0;
  if (wcscmp(W, L"a") <= 0)
    return 0;

  return 1;
}

int test_implicit_compare_with_functions() {

  if (memcmp(A, "a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'memcmp' is called without explicitly comparing result
  // CHECK-FIXES: memcmp(A, "a", 1) != 0)

  if (wmemcmp(W, L"a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'wmemcmp' is called without explicitly comparing result
  // CHECK-FIXES: wmemcmp(W, L"a", 1) != 0)

  if (memicmp(A, "a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'memicmp' is called without explicitly comparing result
  // CHECK-FIXES: memicmp(A, "a", 1) != 0)

  if (_memicmp(A, "a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_memicmp' is called without explicitly comparing result
  // CHECK-FIXES: _memicmp(A, "a", 1) != 0)

  if (_memicmp_l(A, "a", 1, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_memicmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _memicmp_l(A, "a", 1, locale) != 0)

  if (strcmp(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is called without explicitly comparing result
  // CHECK-FIXES: strcmp(A, "a") != 0)

  if (strncmp(A, "a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strncmp' is called without explicitly comparing result
  // CHECK-FIXES: strncmp(A, "a", 1) != 0)

  if (strcasecmp(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcasecmp' is called without explicitly comparing result
  // CHECK-FIXES: strcasecmp(A, "a") != 0)

  if (strncasecmp(A, "a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strncasecmp' is called without explicitly comparing result
  // CHECK-FIXES: strncasecmp(A, "a", 1) != 0)

  if (stricmp(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'stricmp' is called without explicitly comparing result
  // CHECK-FIXES: stricmp(A, "a") != 0)

  if (strcmpi(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmpi' is called without explicitly comparing result
  // CHECK-FIXES: strcmpi(A, "a") != 0)

  if (_stricmp(A, "a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_stricmp' is called without explicitly comparing result
  // CHECK-FIXES: _stricmp(A, "a") != 0)

  if (strnicmp(A, "a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strnicmp' is called without explicitly comparing result
  // CHECK-FIXES: strnicmp(A, "a", 1) != 0)

  if (_strnicmp(A, "a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_strnicmp' is called without explicitly comparing result
  // CHECK-FIXES: _strnicmp(A, "a", 1) != 0)

  if (_stricmp_l(A, "a", locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_stricmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _stricmp_l(A, "a", locale) != 0)

  if (_strnicmp_l(A, "a", 1, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_strnicmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _strnicmp_l(A, "a", 1, locale) != 0)

  if (wcscmp(W, L"a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'wcscmp' is called without explicitly comparing result
  // CHECK-FIXES: wcscmp(W, L"a") != 0)

  if (wcsncmp(W, L"a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'wcsncmp' is called without explicitly comparing result
  // CHECK-FIXES: wcsncmp(W, L"a", 1) != 0)

  if (wcscasecmp(W, L"a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'wcscasecmp' is called without explicitly comparing result
  // CHECK-FIXES: wcscasecmp(W, L"a") != 0)

  if (wcsicmp(W, L"a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'wcsicmp' is called without explicitly comparing result
  // CHECK-FIXES: wcsicmp(W, L"a") != 0)

  if (_wcsicmp(W, L"a"))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_wcsicmp' is called without explicitly comparing result
  // CHECK-FIXES: _wcsicmp(W, L"a") != 0)

  if (_wcsicmp_l(W, L"a", locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_wcsicmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _wcsicmp_l(W, L"a", locale) != 0)

  if (wcsnicmp(W, L"a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'wcsnicmp' is called without explicitly comparing result
  // CHECK-FIXES: wcsnicmp(W, L"a", 1) != 0)

  if (_wcsnicmp(W, L"a", 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_wcsnicmp' is called without explicitly comparing result
  // CHECK-FIXES: _wcsnicmp(W, L"a", 1) != 0)

  if (_wcsnicmp_l(W, L"a", 1, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_wcsnicmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _wcsnicmp_l(W, L"a", 1, locale) != 0)

  if (_mbscmp(U, V))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbscmp' is called without explicitly comparing result
  // CHECK-FIXES: _mbscmp(U, V) != 0)

  if (_mbsncmp(U, V, 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsncmp' is called without explicitly comparing result
  // CHECK-FIXES: _mbsncmp(U, V, 1) != 0)

  if (_mbsnbcmp(U, V, 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsnbcmp' is called without explicitly comparing result
  // CHECK-FIXES: _mbsnbcmp(U, V, 1) != 0)

  if (_mbsnbicmp(U, V, 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsnbicmp' is called without explicitly comparing result
  // CHECK-FIXES: _mbsnbicmp(U, V, 1) != 0)

  if (_mbsicmp(U, V))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsicmp' is called without explicitly comparing result
  // CHECK-FIXES: _mbsicmp(U, V) != 0)

  if (_mbsnicmp(U, V, 1))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsnicmp' is called without explicitly comparing result
  // CHECK-FIXES: _mbsnicmp(U, V, 1) != 0)

  if (_mbscmp_l(U, V, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbscmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _mbscmp_l(U, V, locale) != 0)

  if (_mbsncmp_l(U, V, 1, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsncmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _mbsncmp_l(U, V, 1, locale) != 0)

  if (_mbsicmp_l(U, V, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsicmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _mbsicmp_l(U, V, locale) != 0)

  if (_mbsnicmp_l(U, V, 1, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsnicmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _mbsnicmp_l(U, V, 1, locale) != 0)

  if (_mbsnbcmp_l(U, V, 1, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsnbcmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _mbsnbcmp_l(U, V, 1, locale) != 0)

  if (_mbsnbicmp_l(U, V, 1, locale))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function '_mbsnbicmp_l' is called without explicitly comparing result
  // CHECK-FIXES: _mbsnbicmp_l(U, V, 1, locale) != 0)

  return 1;
}

int strcmp_wrapper1(const char* a, const char* b) {
  return strcmp(a, b);
}

int strcmp_wrapper2(const char* a, const char* b) {
  return (a && b) ? strcmp(a, b) : 0;
}

#define macro_strncmp(s1, s2, n)                                              \
  (__extension__ (__builtin_constant_p (n)                                    \
                  && ((__builtin_constant_p (s1)                              \
                       && strlen (s1) < ((size) (n)))                         \
                      || (__builtin_constant_p (s2)                           \
                          && strlen (s2) < ((size) (n))))                     \
                  ? strcmp (s1, s2) : strncmp (s1, s2, n)))

int strncmp_macro(const char* a, const char* b) {
  if (macro_strncmp(a, b, 4))
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is called without explicitly comparing result

  if (macro_strncmp(a, b, 4) == 2)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' is compared to a suspicious constant

  if (macro_strncmp(a, b, 4) <= .0)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: function 'strcmp' has suspicious implicit cast

  if (macro_strncmp(a, b, 4) + 0)
    return 0;
  // CHECK-MESSAGES: [[@LINE-2]]:7: warning: results of function 'strcmp' used by operator '+'

  return 1;
}
