// Opt-out flag for the built-in stdlib nullable-return list (malloc/fopen/...).
//
// By default (-fnullability-stdlib-annotations, implied), the analysis treats
// known stdlib allocators/lookups as returning nullable pointers, so an
// unchecked dereference of their result warns. With
// -fno-nullability-stdlib-annotations the list is ignored and the same code is
// silent.
//
// On by default (the negative cc1 flag below is what toggles it off):
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -std=c11 -verify=on %s
// RUN: %clang_cc1 -fsyntax-only -fflow-sensitive-nullability -fno-nullability-stdlib-annotations -std=c11 -verify=off %s

typedef __SIZE_TYPE__ size_t;
void *malloc(size_t);
char *getenv(const char *);

// The function must be opted in: annotate the signature so the analysis runs.
char first_byte(int *_Nullable trigger) {
  char *p = malloc(8);
  return *p; // on-warning@+0 {{dereference of nullable pointer}} on-note@+0 {{add a null check}}
}

char env_first(int *_Nullable trigger) {
  char *e = getenv("PATH");
  return *e; // on-warning@+0 {{dereference of nullable pointer}} on-note@+0 {{add a null check}}
}

// Checking the result before deref is always safe, regardless of the flag.
char first_byte_checked(int *_Nullable trigger) {
  char *p = malloc(8);
  if (p)
    return *p;
  return 0;
}

// off-no-diagnostics
