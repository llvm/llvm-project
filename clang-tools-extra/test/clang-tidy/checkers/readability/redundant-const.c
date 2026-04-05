// RUN: %check_clang_tidy -std=c23-or-later %s readability-redundant-const %t

const int n1 = 20;

constexpr const int p1 = 10;
// CHECK-MESSAGES: [[@LINE-1]]:11: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p1 = 10;

const constexpr int p2 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:1: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr int p2 = 20;

static const constexpr int p3 = 20;
// CHECK-MESSAGES: [[@LINE-1]]:8: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: static constexpr int p3 = 20;

// Since constexpr makes only the pointer const, this usage is not redundant.
constexpr const char *n2 = 0;

constexpr const char *const p4 = 0;
// CHECK-MESSAGES: [[@LINE-1]]:23: warning: redundant use of 'const'; 'constexpr' already implies 'const'
// CHECK-FIXES: constexpr const char *p4 = 0;
