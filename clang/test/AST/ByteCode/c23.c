// RUN: %clang_cc1 -std=c23 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c23 -verify=ref,both %s
// RUN: %clang_cc1 -std=c23 -triple=aarch64_be-linux-gnu -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c23 -triple=aarch64_be-linux-gnu -verify=ref,both %s


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

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
#  define LITTLE_END 1
#elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
#  define LITTLE_END 0
#else
#  error "huh?"
#endif

typedef unsigned char u8x4_t __attribute__((vector_size(4)));
constexpr u8x4_t arg1 = (u8x4_t)0xCAFEBABE; // okay
#if LITTLE_END
static_assert(arg1[0] == 190);
static_assert(arg1[1] == 186);
static_assert(arg1[2] == 254);
static_assert(arg1[3] == 202);
#else
static_assert(arg1[0] == 202);
static_assert(arg1[1] == 254);
static_assert(arg1[2] == 186);
static_assert(arg1[3] == 190);
#endif
