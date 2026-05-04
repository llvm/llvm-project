// RUN: %clang_cc1 -triple x86_64-linux %s -std=c2y -verify=expected,both -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux %s -std=c2y -verify=ref,both

// both-no-diagnostics

struct S {
  int x;
  char c;
  float f;
};

#define DECL_BUFFER(Ty, Name) alignas(Ty) unsigned char Name[sizeof(Ty)]

struct T {
  DECL_BUFFER(struct S, buffer);
};

int quorble() {
  DECL_BUFFER(struct T, buffer);
  ((struct S *)((struct T *)buffer)->buffer)->x = 12;
  const struct S *s_ptr = (struct S *)((struct T *)buffer)->buffer;
  return s_ptr->x;
}
