// Some of this should fail in MSVC, but work in clang
// when -fms-compatibility is enabled.
// RUN: %clang -fsyntax-only -fms-compatibility -std=c++20 %s

typedef long LONG;
typedef __int64 LONG_PTR, *PLONG_PTR;

#define FIELD_OFFSET(type, field) ((LONG_PTR)&(((type *)0)->field))

struct S {
  int x;
  int y;
};

constexpr bool cb_eq  = FIELD_OFFSET(S, y) == 4;
constexpr bool cb_ne  = FIELD_OFFSET(S, y) != 0;
constexpr bool cb_lt  = FIELD_OFFSET(S, y) < 8;
constexpr bool cb_le  = FIELD_OFFSET(S, y) <= 4;
constexpr bool cb_gt  = FIELD_OFFSET(S, y) > 0;
constexpr bool cb_ge  = FIELD_OFFSET(S, y) >= 4;
constexpr bool cb_bool = FIELD_OFFSET(S, y);

static_assert(FIELD_OFFSET(S, y) == 4);
static_assert(FIELD_OFFSET(S, y) != 0);
static_assert(FIELD_OFFSET(S, y) < 8);
static_assert(FIELD_OFFSET(S, y) <= 4);
static_assert(FIELD_OFFSET(S, y) > 0);
static_assert(FIELD_OFFSET(S, y) >= 4);
static_assert(FIELD_OFFSET(S, y));


enum E {
  enum_offset_y = FIELD_OFFSET(S, y),
  enum_cmp_y    = FIELD_OFFSET(S, y) == 4
};

int arr_bound[FIELD_OFFSET(S, y)];
int arr_bound_cmp[FIELD_OFFSET(S, y) == 4 ? 1 : -1];

struct BitField {
  int bf1 : FIELD_OFFSET(S, y);
  int bf2 : FIELD_OFFSET(S, y) == 4;
};

template<int N>
struct TplInt {};

template<bool B>
struct TplBool {};

TplInt<FIELD_OFFSET(S, y)> tpl_int;
TplBool<FIELD_OFFSET(S, y) == 4> tpl_bool;

void f() noexcept(FIELD_OFFSET(S, y) == 4) {}

template<class T>
void g() {
  if constexpr (FIELD_OFFSET(S, y) == 4) {
  } else {
  }
}

struct ExplicitCtor {
  explicit(FIELD_OFFSET(S, y) == 4) ExplicitCtor(int) {}
};

alignas(FIELD_OFFSET(S,y)) int __g;

constinit int constinit_offset = FIELD_OFFSET(S, y);
constinit bool constinit_bool = FIELD_OFFSET(S, y) == 4;

constexpr int constexpr_offset = FIELD_OFFSET(S, y);
constexpr int constexpr_cmp_as_int = FIELD_OFFSET(S, y) == 4;
constexpr bool constexpr_bool = FIELD_OFFSET(S, y) == 4;

int switch_test(int v) {
  switch (v) {
  case FIELD_OFFSET(S, y):
    return 1;
  case FIELD_OFFSET(S, x):
    return 2;
  default:
    return 0;
  }
}

template<int N = FIELD_OFFSET(S, y)>
struct DefaultTplInt {};

template<bool B = FIELD_OFFSET(S, y) == 4>
struct DefaultTplBool {};

DefaultTplInt<> default_tpl_int;
DefaultTplBool<> default_tpl_bool;

struct ArrayMember {
  int a[FIELD_OFFSET(S, y)];
};

union U {
  char c;
  int a[FIELD_OFFSET(S, y)];
};

typedef char typedef_arr[FIELD_OFFSET(S, y)];
using using_arr = char[FIELD_OFFSET(S, y)];

constexpr int ternary_offset =
    FIELD_OFFSET(S, y) == 4 ? FIELD_OFFSET(S, y) : -1;

constexpr bool logical_and =
    FIELD_OFFSET(S, y) == 4 && FIELD_OFFSET(S, x) == 0;

constexpr bool logical_or =
    FIELD_OFFSET(S, y) == 4 || FIELD_OFFSET(S, x) == 123;

constexpr bool logical_not =
    !FIELD_OFFSET(S, x);

constexpr int arithmetic_add = FIELD_OFFSET(S, y) + 1;
constexpr int arithmetic_sub = FIELD_OFFSET(S, y) - 1;
constexpr int arithmetic_mul = FIELD_OFFSET(S, y) * 2;
constexpr int arithmetic_div = FIELD_OFFSET(S, y) / 2;
constexpr int arithmetic_mod = FIELD_OFFSET(S, y) % 3;

constexpr int bit_or  = FIELD_OFFSET(S, y) | 1;
constexpr int bit_and = FIELD_OFFSET(S, y) & 7;
constexpr int bit_xor = FIELD_OFFSET(S, y) ^ 1;
constexpr int bit_shl = FIELD_OFFSET(S, y) << 1;
constexpr int bit_shr = FIELD_OFFSET(S, y) >> 1;

constexpr int comma_expr = (0, FIELD_OFFSET(S, y));

constexpr int cast_int = (int)FIELD_OFFSET(S, y);
constexpr long cast_long = (long)FIELD_OFFSET(S, y);
constexpr bool cast_bool = (bool)FIELD_OFFSET(S, y);

template<class T, int N>
struct DependentTpl {};

DependentTpl<S, FIELD_OFFSET(S, y)> dependent_tpl;
