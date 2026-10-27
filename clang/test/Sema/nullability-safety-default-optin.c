// Under the unspecified default a function is analyzed when its body uses an
// annotated declaration or type, or when it is defined inside
// #pragma clang assume_nonnull, even if its own signature has no nullability.
// Unannotated pointers stay unspecified, so only annotated sources (and the
// libc lists) warn in such a function.
//
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -verify %s
// RUN: %clang_cc1 -fsyntax-only -fnullability-safety -std=c11 -Wnullable-to-nonnull-conversion -verify %s

char *getenv(const char *);

struct L { int x; struct L *_Nullable next; struct L *plain; };
struct L *_Nullable split(void);
struct L *lookup(void);
void annotated(struct L *_Nonnull l);
void unannotated(struct L *l);
extern struct L *_Nullable current;
typedef int (*_Nullable maybe_cb)(void);
extern maybe_cb hook;

// An annotated callee opts in an unannotated caller.
void calls_annotated(void) {
  annotated(split()); // expected-warning {{passing nullable pointer to nonnull parameter 'l'}} expected-note {{}}
}

// A _Nullable return dereferenced in an unannotated caller.
int derefs_nullable_return(void) {
  struct L *p = split();
  return p->x; // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}

// A _Nullable global and a _Nullable field.
int derefs_nullable_global(void) {
  return current->x; // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}
int derefs_nullable_field(struct L *l) {
  return l->next->x; // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}

// A typedef'd _Nullable function pointer.
int calls_nullable_hook(void) {
  return hook(); // expected-warning {{calling nullable function pointer}} expected-note {{}}
}

// A local declared with nullability opts in too.
int derefs_nullable_local(struct L *l) {
  struct L *_Nullable q = l->plain;
  return q->x; // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}

// Opted in, but nothing unannotated warns: p, l->plain and lookup() stay
// unspecified.
int opted_in_unannotated_silent(struct L *l) {
  struct L *p = lookup();
  annotated(l->plain);
  annotated(p);
  return p->x + l->plain->x;
}

// Guarded uses stay silent.
int guarded(void) {
  struct L *p = split();
  if (!p)
    return 0;
  annotated(p);
  return p->x;
}

// Nothing annotated is referenced: the function is not analyzed, so even the
// libc nullable-return list does not apply.
int not_opted_in(void) {
  char *s = getenv("X");
  struct L *p = lookup();
  unannotated(p);
  return *s + p->x;
}

// A no-pointer function defined inside the pragma region is analyzed.
#pragma clang assume_nonnull begin
int in_region(void) {
  char *s = getenv("X");
  return *s; // expected-warning {{dereference of nullable pointer}} expected-note {{}}
}
int in_region_checked(void) {
  char *s = getenv("X");
  return s ? *s : 0;
}
#pragma clang assume_nonnull end

// After the region ends, the same body is not analyzed.
int after_region(void) {
  char *s = getenv("X");
  return *s;
}

// A system header's own annotations do not opt a caller in, so a function
// that only calls an annotated libc function keeps its old behavior.
# 1 "sys_annot.h" 1 3
void sys_sort(void *base, int (*_Nonnull cmp)(const void *, const void *));
# 101 "nullability-safety-default-optin.c" 2
int calls_annotated_system_api(void *base) {
  int (*fn)(const void *, const void *) = 0;
  sys_sort(base, fn);
  struct L *p = 0;
  return p->x;
}
