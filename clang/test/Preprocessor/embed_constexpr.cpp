// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%S/Inputs -verify -Wno-c23-extensions
// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%S/Inputs -verify -fexperimental-new-constant-interpreter -Wno-c23-extensions

constexpr int value(int a, int b) {
  return a + b;
}

constexpr int func_call() {
  return value(
#embed <jk.txt>
  );
}

constexpr int init_list_expr() {
  int vals[] = {
#embed <jk.txt>
  };
  return value(vals[0], vals[1]);
}

template <int N, int M>
struct Hurr {
  static constexpr int V1 = N;
  static constexpr int V2 = M;
};

constexpr int template_args() {
  Hurr<
#embed <jk.txt>
  > H;
  return value(H.V1, H.V2);
}

constexpr int ExpectedValue = 'j' + 'k';
static_assert(func_call() == ExpectedValue);
static_assert(init_list_expr() == ExpectedValue);
static_assert(template_args() == ExpectedValue);

static_assert(
#embed <jk.txt> limit(1) suffix(== 'j')
);

int array[
#embed <jk.txt> limit(1)
];
static_assert(sizeof(array) / sizeof(int) == 'j');

constexpr int comma_expr = (
#embed <jk.txt> // expected-warning {{left operand of comma operator has no effect}}
);
static_assert(comma_expr == 'k');

constexpr int comma_expr_init_list{ (
#embed <jk.txt> limit(1)
) };
static_assert(comma_expr_init_list == 'j');

constexpr int paren_init(
#embed <jk.txt> limit(1)
);
static_assert(paren_init == 'j');

struct S {
  const char buffer[2] = {
#embed "jk.txt"
  };
};

constexpr struct S s;
static_assert(s.buffer[1] == 'k');

struct S1 {
  int x, y;
};

struct T {
  int x, y;
  struct S1 s;
};

constexpr struct T t[] = {
#embed <numbers.txt>
};
static_assert(t[0].s.x == '2');

constexpr int func(int i, int) { return i; }
static_assert(
  func(
#embed <jk.txt>
  ) == 'j');

template <int N>
struct ST {};

ST<
#embed <jk.txt> limit(1)
> st;
