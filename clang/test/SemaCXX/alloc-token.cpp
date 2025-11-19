// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -falloc-token-mode=typehash -DMODE_TYPEHASH
// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++23 -fsyntax-only -verify %s -falloc-token-max=2 -DTOKEN_MAX=2

#if !__has_builtin(__builtin_infer_alloc_token)
#error "missing __builtin_infer_alloc_token"
#endif

struct NoPtr {
  int x;
  long y;
};

struct WithPtr {
  int a;
  char *buf;
};

// Check specific known values; these are guaranteed to be stable.
#ifdef MODE_TYPEHASH
static_assert(__builtin_infer_alloc_token(sizeof(int)) == 2689373973731826898ULL);
static_assert(__builtin_infer_alloc_token(sizeof(char*)) == 2250492667400517147ULL);
static_assert(__builtin_infer_alloc_token(sizeof(NoPtr)) == 7465259095297095368ULL);
static_assert(__builtin_infer_alloc_token(sizeof(WithPtr)) == 11898882936532569145ULL);
#elif defined(TOKEN_MAX)
#  if TOKEN_MAX == 2
static_assert(__builtin_infer_alloc_token(sizeof(int)) == 0);
static_assert(__builtin_infer_alloc_token(sizeof(char*)) == 1);
static_assert(__builtin_infer_alloc_token(sizeof(NoPtr)) == 0);
static_assert(__builtin_infer_alloc_token(sizeof(WithPtr)) == 1);
#  else
#    error "unhandled TOKEN_MAX case"
#  endif
#else
static_assert(__builtin_infer_alloc_token(sizeof(int)) == 2689373973731826898ULL);
static_assert(__builtin_infer_alloc_token(sizeof(char*)) == 11473864704255292954ULL);
static_assert(__builtin_infer_alloc_token(sizeof(NoPtr)) == 7465259095297095368ULL);
static_assert(__builtin_infer_alloc_token(sizeof(WithPtr)) == 11898882936532569145ULL);
#endif

// Template function.
template <typename T>
constexpr unsigned long get_token() {
  return __builtin_infer_alloc_token(sizeof(T));
}
static_assert(__builtin_infer_alloc_token(sizeof(int)) == get_token<int>());

// Test complex expressions.
static_assert(__builtin_constant_p(__builtin_infer_alloc_token(sizeof(int))));
static_assert(__builtin_infer_alloc_token(sizeof(NoPtr) * 2, 1) == get_token<NoPtr>());
static_assert(__builtin_infer_alloc_token(1, 4 + sizeof(NoPtr)) == get_token<NoPtr>());
static_assert(__builtin_infer_alloc_token(sizeof(NoPtr) << 8) == get_token<NoPtr>());

// Test usable as a template param.
template <unsigned long ID, typename T>
struct token_for_type {
  static_assert(ID == get_token<T>());
  static constexpr unsigned long value = ID;
};
static_assert(token_for_type<__builtin_infer_alloc_token(sizeof(int)), int>::value == get_token<int>());

template <typename T = void>
void template_test() {
  __builtin_infer_alloc_token(T()); // no error if not instantiated
}

template <typename T>
void negative_template_test() {
  __builtin_infer_alloc_token(T()); // expected-error {{argument may not have 'void' type}}
}

void negative_tests() {
  __builtin_infer_alloc_token(); // expected-error {{too few arguments to function call}}
  __builtin_infer_alloc_token((void)0); // expected-error {{argument may not have 'void' type}}
  negative_template_test<void>(); // expected-note {{in instantiation of function template specialization 'negative_template_test<void>' requested here}}
  constexpr auto inference_fail = __builtin_infer_alloc_token(123); // expected-error {{must be initialized by a constant expression}} \
                                                                    // expected-note {{could not infer allocation type for __builtin_infer_alloc_token}}
}
