// Dereferencing a nullable call result directly warns like dereferencing a
// variable initialized from it: for the libc nullable-return list and for
// functions declared with a _Nullable return.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -fnullability-default=nonnull -verify %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -fnullability-default=nonnull -fno-nullability-libc-nullable-returns -verify=nolibc %s

char *strchr(const char *, int);
struct S { int x; };
struct S *_Nullable find(void);
struct S *plain(void);
void take(struct S *_Nonnull);

// Opted in by its _Nonnull parameter in every mode.
int libc_direct(const char *_Nonnull s) {
  return *strchr(s, ':'); // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}
int libc_via_var(const char *_Nonnull s) {
  char *c = strchr(s, ':');
  return *c; // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}
int libc_subscript(const char *_Nonnull s) {
  return strchr(s, ':')[1]; // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}
int libc_cast(const char *_Nonnull s) {
  return *(const unsigned char *)strchr(s, ':'); // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}

int annotated_direct(void) {
  return find()->x; // expected-warning {{dereference of nullable pointer}} expected-note {{}} nolibc-warning {{dereference of nullable pointer}} nolibc-note {{}}
}
int annotated_star(void) {
  return (*find()).x; // expected-warning {{dereference of nullable pointer}} expected-note {{}} nolibc-warning {{dereference of nullable pointer}} nolibc-note {{}}
}
void annotated_argument(void) {
  take(find()); // expected-warning {{passing nullable pointer to nonnull parameter}} expected-note {{}} nolibc-warning {{passing nullable pointer}} nolibc-note {{}}
}

// Unannotated, non-libc results are not nullable in either mode.
int plain_direct(int *_Nullable trigger) {
  return plain()->x;
}
