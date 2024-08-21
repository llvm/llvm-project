// RUN: %clang_cc1 -std=c23 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c23 -verify=ref,both %s

typedef typeof(nullptr) nullptr_t;

const _Bool inf1 =  (1.0/0.0 == __builtin_inf());
constexpr _Bool inf2 = (1.0/0.0 == __builtin_inf()); // both-error {{must be initialized by a constant expression}} \
                                                     // both-note {{division by zero}}
constexpr _Bool inf3 = __builtin_inf() == __builtin_inf();

/// Used to crash.
struct S {
  int x;
  char c;
  float f;
};

#define DECL_BUFFER(Ty, Name) alignas(Ty) unsigned char Name[sizeof(Ty)]

char bar() {
  DECL_BUFFER(struct S, buffer);
  ((struct S *)buffer)->c = 'a';
  return ((struct S *)buffer)->c;
}


static_assert((nullptr_t){} == 0);
