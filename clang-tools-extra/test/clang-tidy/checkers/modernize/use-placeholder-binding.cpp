// RUN: %check_clang_tidy -std=c++26-or-later %s modernize-use-placeholder-binding %t

struct Pair {
  int first;
  bool second;
};

struct Triple {
  int a, b, c;
};

Pair getPair();
Triple getTriple();
void use(int);

void suppressedBindingIsReplaced() {
  const auto [x, y] = getPair();
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: binding 'y' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: const auto [x, _] = getPair();
  (void)y;
  // CHECK-FIXES-NOT: (void)y;
  use(x);
}

void ifInitBindingIsReplaced() {
  if (const auto [it, inserted] = getPair(); inserted) {
    // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: binding 'it' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
    // CHECK-FIXES: if (const auto [_, inserted] = getPair(); inserted) {
    (void)it;
    // CHECK-FIXES-NOT: (void)it;
  }
}

void bothBindingsUsed() {
  const auto [x, y] = getPair();
  use(x);
  if (y)
    use(x);
}

void noSuppressionCast() {
  const auto [x, y] = getPair();
  use(x);
}

void alreadyPlaceholder() {
  const auto [x, _] = getPair();
  use(x);
}

void bothBindingsSuppressed() {
  auto [x, y] = getPair();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: binding 'x' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: binding 'y' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: auto [_, _] = getPair();
  (void)x;
  // CHECK-FIXES-NOT: (void)x;
  (void)y;
  // CHECK-FIXES-NOT: (void)y;
}

void threeBindings() {
  auto [a, b, c] = getTriple();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: binding 'b' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-MESSAGES: :[[@LINE-2]]:15: warning: binding 'c' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: auto [a, _, _] = getTriple();
  (void)b;
  (void)c;
  use(a);
}

void referenceBinding() {
  Pair p{};
  auto &[x, y] = p;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: binding 'y' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: auto &[x, _] = p;
  (void)y;
  use(x);
}

struct Item {
  int key;
  int value;
};

void forRangeBindingIsReplaced() {
  Item items[1] = {};
  for (auto [k, v] : items) {
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: binding 'v' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
    // CHECK-FIXES: for (auto [k, _] : items) {
    (void)v;
    // CHECK-FIXES-NOT: (void)v;
    use(k);
  }
}

void switchCaseBindingIsReplaced() {
  switch (auto [a, b] = getPair(); a) {
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: binding 'b' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: switch (auto [a, _] = getPair(); a) {
  case 0:
    (void)b;
    // CHECK-FIXES-NOT: (void)b;
    break;
  }
}

void defaultCaseSoleSubstatementIsReplaced() {
  switch (auto [a, b] = getPair(); a) {
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: binding 'b' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: switch (auto [a, _] = getPair(); a) {
  default:
    (void)b;
    // CHECK-FIXES-NOT: (void)b;
  }
}

void lambdaCaptureIsUnaffected() {
  auto [x, y] = getPair();
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: binding 'y' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: auto [x, _] = getPair();
  (void)y;
  // CHECK-FIXES-NOT: (void)y;
  auto l = [x] { use(x); };
  l();
}

#define SUPPRESS(x) (void)(x)
void macroSuppressionIsNotDiagnosed() {
  auto [x, y] = getPair();
  SUPPRESS(y);
  use(x);
}

void multipleDecompositionsInSameScope() {
  auto [x1, y1] = getPair();
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: binding 'y1' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: auto [x1, _] = getPair();
  (void)y1;
  // CHECK-FIXES-NOT: (void)y1;
  use(x1);

  auto [x2, y2] = getPair();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: binding 'x2' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: auto [_, y2] = getPair();
  (void)x2;
  // CHECK-FIXES-NOT: (void)x2;
  use(y2 ? 1 : 0);
}

Pair &&getPairRef();

void rvalueReferenceBinding() {
  auto &&[x, y] = getPairRef();
  // CHECK-MESSAGES: :[[@LINE-1]]:14: warning: binding 'y' is only used to suppress an unused variable warning; use a placeholder '_' instead [modernize-use-placeholder-binding]
  // CHECK-FIXES: auto &&[x, _] = getPairRef();
  (void)y;
  // CHECK-FIXES-NOT: (void)y;
  use(x);
  // CHECK-FIXES: use(x);
}

void staticBindingIsNotDiagnosed() {
  static auto [x1, y1] = getPair();
  (void)y1;
  static auto [x2, y2] = getPair();
  (void)y2;
  use(x1);
  use(x2);
}

void threadLocalBindingIsNotDiagnosed() {
  thread_local auto [x, y] = getPair();
  (void)y;
  use(x);
}
