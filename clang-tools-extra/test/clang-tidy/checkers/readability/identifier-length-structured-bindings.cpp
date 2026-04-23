
// RUN: %check_clang_tidy -std=c++17-or-later %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {}}' \
// RUN: -- -fexceptions

struct aggregate {
  int first;
  double middle;
  bool last;
};

aggregate get_data();

void shouldWarn() {
  auto [f, m, l] = get_data();
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: binding variable name 'f' is too short, expected at least 2 characters [readability-identifier-length]
  // CHECK-MESSAGES: :[[@LINE-2]]:12: warning: binding variable name 'm' is too short, expected at least 2 characters [readability-identifier-length]
  // CHECK-MESSAGES: :[[@LINE-3]]:15: warning: binding variable name 'l' is too short, expected at least 2 characters [readability-identifier-length]
}

void shouldNotWarn() {
  auto [_, mid, last] = get_data();
}
