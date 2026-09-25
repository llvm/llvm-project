// RUN: %clang_cc1 -triple x86_64-linux -verify=both,expected %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux -verify=both,ref      %s



static_assert("foo"[0] == 'f');
static_assert("foo"[1] == 'o');
static_assert("foo"[2] == 'o');
static_assert("foo"[3] == '\0');

static_assert(+"foo" == +"foo"); // both-error {{not an integral constant expression}} \
                                 // both-note {{comparison of addresses of potentially overlapping literals}}

static_assert("foo"[4] == '\0'); // both-error {{not an integral constant expression}} \
                                 // both-note {{read of dereferenced one-past-the-end pointer}}

static_assert("foo"[5] == '\0'); // both-error {{not an integral constant expression}} \
                                 // both-note {{cannot refer to element 5 of array of 4 elements}}

constexpr const wchar_t *wide = L"bar";
static_assert(wide[0] == L'b', "");

constexpr const char32_t *u32 = U"abc";
static_assert(u32[1] == U'b', "");

constexpr int testMemcpy() {
  char s[5] = {0, 0, 0, 0, 0};
  __builtin_memcpy(s, "abcd", 5);
  return s[0] == 'a';
}
static_assert(testMemcpy() == 1, "");

constexpr const auto *wp = L"abc";
static_assert(&wp[2] - &wp[0] == 2);


constexpr int checkMemcpy() {
  char a[3] = {};

  __builtin_memcpy(a, &"abcdef"[3], 3);
  return __builtin_strncmp(a, "def", 3) == 0;
}
static_assert(checkMemcpy());



