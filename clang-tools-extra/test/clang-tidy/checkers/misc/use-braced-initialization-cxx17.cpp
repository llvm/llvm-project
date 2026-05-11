// RUN: %check_clang_tidy -std=c++17-or-later %s misc-use-braced-initialization %t -- --fix-notes

struct Agg {
  int a, b;
};

void structured_binding_paren() {
  auto [a, b](Agg{1, 2});
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-FIXES: auto [a, b]{Agg{1, 2}};
}

void structured_binding_braced() {
  auto [a, b] = Agg{1, 2};
}

void structured_binding_paren_call() {
  Agg make_agg();
  auto [a, b](make_agg());
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-FIXES: auto [a, b]{make_agg()};
}

void structured_binding_paren_copy(const Agg &g) {
  auto [a, b](g);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-FIXES: auto [a, b]{g};
}

void structured_binding_paren_ref(Agg &g) {
  auto &[a, b](g);
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: use braced initialization
  // CHECK-FIXES: auto &[a, b]{g};
}

void structured_binding_paren_array() {
  int arr[2] = {1, 2};
  auto [a, b](arr);
  // CHECK-MESSAGES: :[[@LINE-1]]:8: warning: use braced initialization
  // CHECK-FIXES: auto [a, b]{arr};
}
