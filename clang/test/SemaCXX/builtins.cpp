// RUN: %clang_cc1 %s -fsyntax-only -verify
typedef const struct __CFString * CFStringRef;
#define CFSTR __builtin___CFStringMakeConstantString

void f() {
  (void)CFStringRef(CFSTR("Hello"));
}

void a() { __builtin_va_list x, y; ::__builtin_va_copy(x, y); }

// <rdar://problem/10063539>
template<int (*Compare)(const char *s1, const char *s2)>
int equal(const char *s1, const char *s2) {
  return Compare(s1, s2) == 0;
}
// FIXME: Our error recovery here sucks
template int equal<&__builtin_strcmp>(const char*, const char*); // expected-error {{builtin functions must be directly called}} expected-error {{expected unqualified-id}} expected-error {{expected ')'}} expected-note {{to match this '('}}

// PR13195
void f2() {
  __builtin_isnan; // expected-error {{builtin functions must be directly called}}
}
