// RUN: %check_clang_tidy %s cppcoreguidelines-prefer-member-initializer,modernize-use-default-member-init %t -- \
// RUN: -config="{CheckOptions: [{key: modernize-use-default-member-init.UseAssignment, value: true}]}"
// RUN: %check_clang_tidy %s cppcoreguidelines-prefer-member-initializer,modernize-use-default-member-init %t -- \
// RUN: -config="{CheckOptions: [{key: modernize-use-default-member-init.UseAssignment, value: false}, \
// RUN:                          {key: cppcoreguidelines-prefer-member-initializer.UseAssignment, value: true}]}"

class Simple1 {
  int n;
  // CHECK-FIXES: int n = 0;
  double x;
  // CHECK-FIXES: double x = 0.0;

public:
  Simple1() {
    n = 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in an in-class default member initializer [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    x = 0.0;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x' should be initialized in an in-class default member initializer [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  Simple1(int nn, double xx) {
    // CHECK-FIXES: Simple1(int nn, double xx) : n(nn), x(xx) {
    n = nn;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'n' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
    x = xx;
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: 'x' should be initialized in a member initializer of the constructor [cppcoreguidelines-prefer-member-initializer]
    // CHECK-FIXES: {{^\ *$}}
  }

  ~Simple1() = default;
};
