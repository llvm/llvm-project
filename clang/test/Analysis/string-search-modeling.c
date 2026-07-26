// RUN: %clang_analyze_cc1 -verify %s \
// RUN:   -analyzer-checker=core,unix \
// RUN:   -analyzer-checker=debug.ExprInspection \
// RUN:   -analyzer-config eagerly-assume=false

typedef __SIZE_TYPE__ size_t;
void *malloc(size_t size);
void free(void *p);
void *memcpy(void *dest, const void *src, size_t n);
char *strchr(const char *s, int c);
char *strrchr(const char *s, int c);
char *strstr(const char *haystack, const char *needle);
char *strpbrk(const char *s, const char *accept);
void *memchr(const void *s, int c, size_t n);
char *strchrnul(const char *s, int c);

void clang_analyzer_eval(int);

//===----------------------------------------------------------------------===//
// Check for stack address escapes.
//===----------------------------------------------------------------------===//

char *returns_stack_strchr(void) {
  char buf[8] = "abc";
  return strchr(buf, 'b');
  // expected-warning@-1 {{Address of stack memory associated with local variable 'buf' returned to caller}}
}

char *returns_stack_strrchr(void) {
  char buf[8] = "abc";
  return strrchr(buf, 'b');
  // expected-warning@-1 {{Address of stack memory associated with local variable 'buf' returned to caller}}
}

char *returns_stack_strstr(void) {
  char buf[8] = "abc";
  return strstr(buf, "b");
  // expected-warning@-1 {{Address of stack memory associated with local variable 'buf' returned to caller}}
}

char *returns_stack_strpbrk(void) {
  char buf[8] = "abc";
  return strpbrk(buf, "b");
  // expected-warning@-1 {{Address of stack memory associated with local variable 'buf' returned to caller}}
}

void *returns_stack_memchr(void) {
  char buf[8] = "abc";
  return memchr(buf, 'b', sizeof buf);
  // expected-warning@-1 {{Address of stack memory associated with local variable 'buf' returned to caller}}
}

char *returns_stack_strchrnul(void) {
  char buf[8] = "abc";
  return strchrnul(buf, 'b');
  // expected-warning@-1 {{Address of stack memory associated with local variable 'buf' returned to caller}}
}

char *forwards_param(char *p) {
  return strchr(p, 'b'); // no-warning
}

char *returns_local_static(void) {
  extern char g[8];
  return strchr(g, 'b'); // no-warning
}

//===----------------------------------------------------------------------===//
// unix.cstring.NullArg: the source pointer must be non-null.
//===----------------------------------------------------------------------===//

void null_source_strchr(int c) {
  strchr(0, c);
  // expected-warning@-1 {{Null pointer passed as 1st argument to strchr()}}
}

void null_source_strrchr(int c) {
  strrchr(0, c);
  // expected-warning@-1 {{Null pointer passed as 1st argument to strrchr()}}
}

void null_source_memchr(int c) {
  memchr(0, c, 4);
  // expected-warning@-1 {{Null pointer passed as 1st argument to memchr()}}
}

void null_source_strstr(void) {
  strstr(0, "x");
  // expected-warning@-1 {{Null pointer passed as 1st argument to strstr()}}
}

void null_source_strpbrk(void) {
  strpbrk(0, "x");
  // expected-warning@-1 {{Null pointer passed as 1st argument to strpbrk()}}
}

void null_source_strchrnul(int c) {
  strchrnul(0, c);
  // expected-warning@-1 {{Null pointer passed as 1st argument to strchrnul()}}
}

//===----------------------------------------------------------------------===//
// State split: result == NULL on one branch, in the source on the other.
//===----------------------------------------------------------------------===//

// Both branches are reachable; the verifier matches the two values set-wise.
void state_split(const char *p) {
  clang_analyzer_eval(strchr(p, 'b') == 0);    // expected-warning {{TRUE}} expected-warning {{FALSE}}
  clang_analyzer_eval(strrchr(p, 'b') == 0);   // expected-warning {{TRUE}} expected-warning {{FALSE}}
  clang_analyzer_eval(strstr(p, "x") == 0);    // expected-warning {{TRUE}} expected-warning {{FALSE}}
  clang_analyzer_eval(strpbrk(p, "x") == 0);   // expected-warning {{TRUE}} expected-warning {{FALSE}}
  clang_analyzer_eval(memchr(p, 'b', 4) == 0); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}

// strchrnul does not split: it never returns NULL at runtime.
void strchrnul_is_nonnull(const char *p) {
  clang_analyzer_eval(strchrnul(p, 'b') == 0); // expected-warning {{FALSE}}
}

// On the "found" branch the result aliases the source region, but the offset
// is opaque, so equality with in-source pointers is UNKNOWN.
void found_branch_offset_is_opaque(const char *p) {
  char *q = strchr(p, 'b');
  if (!q) return; // constrain to "found" branch
  clang_analyzer_eval(q == p);     // expected-warning {{UNKNOWN}}
  clang_analyzer_eval(q == p + 1); // expected-warning {{UNKNOWN}}
}

void resulting_ptr_shares_provenance_with_src(int rng, char *opaque) {
  if (rng == 10) {
    char *q = strchr("abcd", 'b');
    free(q); // expected-warning {{Argument to 'free()' is the address of a global variable, which is not memory allocated by 'malloc()'}}
    return;
  }

  if (rng == 20) {
    char *q = strchr(opaque, 'b');
    free(q); // ok
    return;
  }

  if (rng == 30) {
    char *q = strchr(opaque, 'b');
    free(q); // Notionally releases 'opaque'.
    free(opaque); // expected-warning {{Attempt to release already released memory}}
    return;
  }
}

//===----------------------------------------------------------------------===//
// core.NullDereference:
// A returned pointer used without a NULL check is flagged on the NULL branch.
//===----------------------------------------------------------------------===//

void deref_unchecked(const char *s) {
  char *p = strchr(s, 'b');
  *p = 'X'; // expected-warning {{Dereference of null pointer}}
}

void deref_after_check(const char *s) {
  char *p = strchr(s, 'b');
  if (p) {
    *p = 'X'; // no-warning
  }
}

//===----------------------------------------------------------------------===//
// Calling these functions does not invalidate unrelated memory.
//===----------------------------------------------------------------------===//

int global_unmodified;
void no_invalidation_of_globals(const char *p) {
  int local_unmodified = 10;
  global_unmodified = 20;
  (void)strchr(p, 'b');
  clang_analyzer_eval(local_unmodified == 10);  // expected-warning {{TRUE}}
  clang_analyzer_eval(global_unmodified == 20); // expected-warning {{TRUE}}
}

//===----------------------------------------------------------------------===//
// When both arguments are compile-time constants, only the correct branch is
// taken: found when the target exists, null when it does not.
// See: https://github.com/llvm/llvm-project/issues/209905
//===----------------------------------------------------------------------===//

// --- strchr / strrchr: target character IS in the literal ---
const char *test_strrchr_const_no_fp(void) {
  // This is the original reproducer from #209905.
  return strrchr("/foo/bar.c", '/') ? strrchr("/foo/bar.c", '/') + 1 : "/foo/bar.c"; // no-warning
}

void test_strchr_const_found(void) {
  clang_analyzer_eval(strchr("/foo/bar.c", '/') == 0); // expected-warning {{FALSE}}
}

void test_strrchr_const_found(void) {
  clang_analyzer_eval(strrchr("/foo/bar.c", '/') == 0); // expected-warning {{FALSE}}
}

// --- strchr / strrchr: target character is NOT in the literal ---
void test_strchr_const_not_found(void) {
  clang_analyzer_eval(strchr("hello", 'z') == 0); // expected-warning {{TRUE}}
}

void test_strrchr_const_not_found(void) {
  clang_analyzer_eval(strrchr("hello", 'z') == 0); // expected-warning {{TRUE}}
}

// --- memchr: character within bounds ---
void test_memchr_const_found(void) {
  clang_analyzer_eval(memchr("abcdef", 'c', 6) == 0); // expected-warning {{FALSE}}
}

// --- memchr: character beyond the specified length ---
void test_memchr_const_not_in_range(void) {
  clang_analyzer_eval(memchr("abcdef", 'f', 3) == 0); // expected-warning {{TRUE}}
}

// --- strstr: needle IS a substring ---
void test_strstr_const_found(void) {
  clang_analyzer_eval(strstr("hello world", "world") == 0); // expected-warning {{FALSE}}
}

// --- strstr: needle is NOT a substring ---
void test_strstr_const_not_found(void) {
  clang_analyzer_eval(strstr("hello world", "xyz") == 0); // expected-warning {{TRUE}}
}

// --- strpbrk: accept set has a match ---
void test_strpbrk_const_found(void) {
  clang_analyzer_eval(strpbrk("hello", "aeiou") == 0); // expected-warning {{FALSE}}
}

// --- strpbrk: no character from accept set in source ---
void test_strpbrk_const_not_found(void) {
  clang_analyzer_eval(strpbrk("hello", "xyz") == 0); // expected-warning {{TRUE}}
}

// --- Non-constant source: both branches must still exist ---
void test_strchr_non_const_source(const char *p) {
  clang_analyzer_eval(strchr(p, '/') == 0); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}

// --- Various constant source forms: const, static const, #define, __FILE__ ---
static const char static_const_path[] = "/usr/local/bin/tool";

void test_strchr_static_const(void) {
  clang_analyzer_eval(strchr(static_const_path, '/') == 0); // expected-warning {{FALSE}}
}

const char global_const_path[] = "/etc/config";

void test_strrchr_global_const(void) {
  clang_analyzer_eval(strrchr(global_const_path, '/') == 0); // expected-warning {{FALSE}}
}

#define FIXED_PATH "/home/user/project/file.c"

void test_strchr_define(void) {
  clang_analyzer_eval(strchr(FIXED_PATH, '/') == 0); // expected-warning {{FALSE}}
}

#define MY_FILE_BASENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

void test_file_basename_macro(void) {
  const char *base = MY_FILE_BASENAME; // no-warning
  (void)base;
}

#define PREFIX "module"
#define SUFFIX "_handler"
// Adjacent string literal concatenation (the realistic preprocessor pattern):
#define MODULE_PATH "/opt/" PREFIX "/" SUFFIX ".so"

void test_strchr_concatenated_define(void) {
  clang_analyzer_eval(strchr(MODULE_PATH, '/') == 0); // expected-warning {{FALSE}}
}

// --- Character argument via #define ---
#define SEPARATOR '/'

void test_strchr_define_char(void) {
  clang_analyzer_eval(strchr("/foo/bar", SEPARATOR) == 0); // expected-warning {{FALSE}}
}

#define SEARCH_CHAR 'x'

void test_strchr_define_char_not_found(void) {
  clang_analyzer_eval(strchr("/foo/bar", SEARCH_CHAR) == 0); // expected-warning {{TRUE}}
}

// --- Edge cases: null character '\0' ---
void test_strchr_null_char_always_found(void) {
  // strchr(s, '\0') always finds the terminator.
  clang_analyzer_eval(strchr("hello", '\0') == 0); // expected-warning {{FALSE}}
}

void test_strrchr_null_char_always_found(void) {
  clang_analyzer_eval(strrchr("hello", '\0') == 0); // expected-warning {{FALSE}}
}

void test_memchr_null_char_within_bounds(void) {
  // "abc" has terminator at index 3; searching 4 bytes includes it.
  clang_analyzer_eval(memchr("abc", '\0', 4) == 0); // expected-warning {{FALSE}}
}

void test_memchr_null_char_out_of_bounds(void) {
  // "abc" has terminator at index 3; searching only 3 bytes misses it.
  clang_analyzer_eval(memchr("abc", '\0', 3) == 0); // expected-warning {{TRUE}}
}

// --- Edge cases: empty strings ---
void test_strchr_empty_haystack(void) {
  // Empty string only contains '\0'; '/' is not there.
  clang_analyzer_eval(strchr("", '/') == 0); // expected-warning {{TRUE}}
}

void test_strchr_empty_haystack_null_char(void) {
  // strchr("", '\0') finds the terminator.
  clang_analyzer_eval(strchr("", '\0') == 0); // expected-warning {{FALSE}}
}

void test_strstr_empty_needle(void) {
  // strstr(s, "") always returns s.
  clang_analyzer_eval(strstr("hello", "") == 0); // expected-warning {{FALSE}}
}

void test_strpbrk_empty_accept(void) {
  // strpbrk(s, "") never matches.
  clang_analyzer_eval(strpbrk("hello", "") == 0); // expected-warning {{TRUE}}
}

//===----------------------------------------------------------------------===//
// Verify exact pointer offsets when both arguments are compile-time constants.
// The enhanced modeling returns Src + concrete_offset rather than a symbolic
// offset, enabling precise pointer arithmetic downstream.
//===----------------------------------------------------------------------===//

// --- strchr: returns pointer to first occurrence ---
void test_strchr_exact_offset(void) {
  const char *s = "/foo/bar.c";
  // '/' first appears at index 0.
  clang_analyzer_eval(strchr(s, '/') == s); // expected-warning {{TRUE}}
}

// --- strrchr: returns pointer to last occurrence ---
void test_strrchr_exact_offset(void) {
  const char *s = "/foo/bar.c";
  // '/' last appears at index 4.
  clang_analyzer_eval(strrchr(s, '/') == s + 4); // expected-warning {{TRUE}}
}

// --- strrchr + 1: the FILE_BASENAME pattern ---
void test_strrchr_plus_one(void) {
  const char *s = "/foo/bar.c";
  const char *base = strrchr(s, '/') + 1;
  // Should point to 'b' at index 5.
  clang_analyzer_eval(base == s + 5); // expected-warning {{TRUE}}
}

// --- strchr with null terminator: points to end of string ---
void test_strchr_null_terminator_offset(void) {
  const char *s = "hello";
  // strchr(s, '\0') returns pointer to the null terminator at index 5.
  clang_analyzer_eval(strchr(s, '\0') == s + 5); // expected-warning {{TRUE}}
}

// --- strstr: returns pointer to first substring match ---
void test_strstr_exact_offset(void) {
  const char *s = "hello world";
  // "world" starts at index 6.
  clang_analyzer_eval(strstr(s, "world") == s + 6); // expected-warning {{TRUE}}
}

// --- strstr with empty needle: returns the source pointer ---
void test_strstr_empty_needle_offset(void) {
  const char *s = "hello";
  clang_analyzer_eval(strstr(s, "") == s); // expected-warning {{TRUE}}
}

// --- strpbrk: returns pointer to first matching character ---
void test_strpbrk_exact_offset(void) {
  const char *s = "hello";
  // First vowel 'e' is at index 1.
  clang_analyzer_eval(strpbrk(s, "aeiou") == s + 1); // expected-warning {{TRUE}}
}

// --- memchr: returns pointer to character within bounds ---
void test_memchr_exact_offset(void) {
  const char *s = "abcdef";
  // 'c' is at index 2.
  clang_analyzer_eval(memchr(s, 'c', 6) == s + 2); // expected-warning {{TRUE}}
}

// --- memchr with embedded null characters ---
void test_memchr_embedded_null(void) {
  // String literal "ab\0cd" has a null at index 2, then 'c' at 3, 'd' at 4,
  // and the implicit terminator at index 5.
  const char *s = "ab\0cd";
  // memchr searching 5 bytes finds the first '\0' at index 2.
  clang_analyzer_eval(memchr(s, '\0', 5) == s + 2); // expected-warning {{TRUE}}
}

void test_memchr_second_segment_after_null(void) {
  const char *s = "ab\0cd";
  // 'c' is at index 3; searching 5 bytes should find it.
  clang_analyzer_eval(memchr(s, 'c', 5) == s + 3); // expected-warning {{TRUE}}
}

void test_memchr_char_before_null_boundary(void) {
  const char *s = "ab\0cd";
  // 'b' is at index 1; searching only 2 bytes still finds it.
  clang_analyzer_eval(memchr(s, 'b', 2) == s + 1); // expected-warning {{TRUE}}
}

void test_memchr_char_hidden_by_short_len(void) {
  const char *s = "ab\0cd";
  // 'c' is at index 3 but searching only 3 bytes (indices 0-2) misses it.
  clang_analyzer_eval(memchr(s, 'c', 3) == 0); // expected-warning {{TRUE}}
}

// --- strstr/strpbrk with embedded null characters ---
void test_strstr_hidden_by_null(void) {
  const char *s = "ab\0cd";
  // C strstr stops at the first null; "cd" is unreachable.
  clang_analyzer_eval(strstr(s, "cd") == 0); // expected-warning {{TRUE}}
}

void test_strpbrk_hidden_by_null(void) {
  const char *s = "ab\0cd";
  // C strpbrk stops at the first null; 'c' is unreachable.
  clang_analyzer_eval(strpbrk(s, "cd") == 0); // expected-warning {{TRUE}}
}

void test_strstr_before_null(void) {
  const char *s = "1ab\0cd";
  // "ab" is before the null, so strstr finds it at offset 1.
  clang_analyzer_eval(strstr(s, "ab") == s + 1); // expected-warning {{TRUE}}
}

void test_strpbrk_before_null(void) {
  const char *s = "ab\0cd";
  // 'a' is before the null, so strpbrk finds it.
  clang_analyzer_eval(strpbrk(s, "a") == s); // expected-warning {{TRUE}}
}

// --- memchr with pointer past embedded null ---
void test_memchr_pointer_past_null(void) {
  const char *s = "1ab\0cdf\0qwrt";
  // Starting from s+3 ("\0cdf\0qwrt"), search for 'd' in 4 bytes.
  clang_analyzer_eval(memchr(s + 3, 'd', 4) == s + 5); // expected-warning {{TRUE}}
}

// --- Second argument with pointer offset ---
void test_strstr_needle_with_offset(void) {
  const char *needles = "xxworld";
  // needles + 2 is "world"; strstr finds it at index 6.
  clang_analyzer_eval(strstr("hello world", needles + 2) == 0); // expected-warning {{FALSE}}
}

void test_strpbrk_accept_with_offset(void) {
  const char *chars = "xxaeiou";
  // chars + 2 is "aeiou"; first vowel 'e' in "hello" is at index 1.
  clang_analyzer_eval(strpbrk("hello", chars + 2) == 0); // expected-warning {{FALSE}}
}

// --- Both arguments with pointer offsets ---
void test_strstr_both_offsets(void) {
  const char *s = "XXhello world";
  const char *needles = "xxworld";
  // s+2 is "hello world", needles+2 is "world"; found at offset 6 from s+2.
  clang_analyzer_eval(strstr(s + 2, needles + 2) == s + 8); // expected-warning {{TRUE}}
}

void test_strpbrk_both_offsets(void) {
  const char *s = "XXhello";
  const char *chars = "xxaeiou";
  // s+2 is "hello", chars+2 is "aeiou"; first vowel 'e' at offset 1 from s+2.
  clang_analyzer_eval(strpbrk(s + 2, chars + 2) == s + 3); // expected-warning {{TRUE}}
}

// --- strchrnul: always returns non-null (pointer to found char or terminator)
void test_strchrnul_not_found_points_to_terminator(void) {
  const char *s = "hello";
  // strchrnul(s, 'z') should return s + 5 (the null terminator).
  clang_analyzer_eval(strchrnul(s, 'z') == s + 5); // expected-warning {{TRUE}}
}

void test_strchrnul_found_negative(void) {
  const char *s = "xxuhello";
  const char *ss = s + 3;
  // 'u' is at index 1 in "xuhello", so strchrnul(ss-2, 'u') == s + 1 + 1 == s + 2.
  clang_analyzer_eval(strchrnul(ss - 2, 'u') == s + 2); // expected-warning {{TRUE}}
}

void test_strchrnul_found_with_offset(void) {
  const char *s = "xxxhello";
  const char *ss = s + 1;
  // ss is "xxhello", 'l' is at index 4 in "xxhello", so ss + 4 == s + 5.
  clang_analyzer_eval(strchrnul(ss, 'l') == s + 5); // expected-warning {{TRUE}}
}

// --- Embedded null in the needle/accept set argument ---
void test_strstr_needle_embedded_null(void) {
  // Needle "cd\0e" is truncated to "cd" by C semantics; "cd" is in "abcd".
  clang_analyzer_eval(strstr("abcd", "cd\0e") == 0); // expected-warning {{FALSE}}
}

void test_strpbrk_accept_embedded_null(void) {
  // Accept "a\0b" is truncated to "a"; 'a' is not in "xb", so result is null.
  clang_analyzer_eval(strpbrk("xb", "a\0b") == 0); // expected-warning {{TRUE}}
}

// --- Wide string cast to char*: resolved via raw bytes ---
const __CHAR16_TYPE__ wide_str_global[] = u"abc";
void test_strchr_wide_string_global(void) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // LE: bytes are 'a',0,'b',0,... — CStr is "a", strchr finds 'a' at offset 0.
  clang_analyzer_eval(strchr((const char *)wide_str_global, 'a') == (const char *)wide_str_global); // expected-warning {{TRUE}}
#else
  // BE: bytes are 0,'a',0,'b',... — first byte is null, CStr is empty.
  clang_analyzer_eval(strchr((const char *)wide_str_global, 'a') == 0); // expected-warning {{TRUE}}
#endif
}

void test_strchr_wide_string_local(void) {
  const __CHAR16_TYPE__ w[] = u"abc";
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // LE: same as global — finds 'a' at offset 0.
  clang_analyzer_eval(strchr((const char *)w, 'a') == (const char *)w); // expected-warning {{TRUE}}
#else
  // BE: first byte is null, CStr is empty.
  clang_analyzer_eval(strchr((const char *)w, 'a') == 0); // expected-warning {{TRUE}}
#endif
}

// --- Wide string with 4-byte characters (UTF-32) ---
const __CHAR32_TYPE__ wide32_global[] = U"abc";
void test_strchr_wide32_string(void) {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // LE: U"abc" bytes are 'a',0,0,0,'b',0,0,0,...  CStr is "a".
  clang_analyzer_eval(strchr((const char *)wide32_global, 'a') == (const char *)wide32_global); // expected-warning {{TRUE}}
#else
  // BE: bytes are 0,0,0,'a',... — first byte is null, CStr is empty.
  clang_analyzer_eval(strchr((const char *)wide32_global, 'a') == 0); // expected-warning {{TRUE}}
#endif
}

// --- Wide string as needle argument ---
const __CHAR16_TYPE__ wide_needle[] = u"lo";
void test_strstr_wide_needle(void) {
  const char *s = "hello";
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
  // LE: u"lo" bytes are 'l',0,'o',0,0,0 — getCStr gives "l".
  // strstr("hello", "l") finds 'l' at offset 2.
  clang_analyzer_eval(strstr(s, (const char *)wide_needle) == s + 2); // expected-warning {{TRUE}}
#else
  // BE: bytes are 0,'l',0,'o',... — CStr is empty, strstr returns haystack.
  clang_analyzer_eval(strstr(s, (const char *)wide_needle) == s); // expected-warning {{TRUE}}
#endif
}

// --- Struct cast to char* ---
struct FourChars {
  char a, b, c, d;
};
void test_strchr_struct_cast(void) {
  // TODO: resolve const struct initializers (no padding between char members).
  struct FourChars s = {'h', 'e', 'l', 'l'};
  clang_analyzer_eval(strchr((const char *)&s, 'e') == 0); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}

// --- Union with wide and narrow access ---
union CharUnion {
  __CHAR16_TYPE__ w[4];
  char c[8];
};
const union CharUnion cu = { .w = u"abc" };
void test_strchr_union_narrow_access(void) {
  // TODO: resolve union members with known initializers.
  clang_analyzer_eval(strchr(cu.c, 'a') == 0); // expected-warning {{TRUE}} expected-warning {{FALSE}}
}
