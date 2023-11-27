// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%S/Inputs -verify -Wno-c23-extensions
// RUN: %clang_cc1 %s -fsyntax-only --embed-dir=%S/Inputs -verify=fixme -fexperimental-new-constant-interpreter -Wno-c23-extensions
// expected-no-diagnostics

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
static_assert(template_args() == ExpectedValue); // fixme-error {{static assertion expression is not an integral constant expression}}

static_assert(
#embed <jk.txt> limit(1) suffix(== 'j')
);

int array[
#embed <jk.txt> limit(1)
];
static_assert(sizeof(array) / sizeof(int) == 'j');

constexpr int comma_expr = (
#embed <jk.txt>
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
