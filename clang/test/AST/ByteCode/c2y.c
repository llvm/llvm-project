// RUN: %clang_cc1 -triple x86_64-linux %s -std=c2y -verify=expected,both -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux %s -std=c2y -verify=ref,both

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


__UINT_LEAST32_TYPE__ stdc_load8_aligned_leu32(const unsigned char *);
alignas(8) unsigned char buf[4] = {};
__UINT_LEAST32_TYPE__ fail_load8 = stdc_load8_aligned_leu32(buf + 4); // both-error {{not a compile-time constant}}


