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
