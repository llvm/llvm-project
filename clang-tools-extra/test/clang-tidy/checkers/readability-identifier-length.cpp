// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: \
// RUN:  [{key: readability-identifier-length.IgnoredVariableNames, value: "^[xy]$"}]}' \
// RUN: -- -fexceptions

struct myexcept {
  int val;
};

struct simpleexcept {
  int other;
};

void doIt();

void tooShortVariableNames(int z)
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: parameter name 'z' is too short, expected at least 3 characters [readability-identifier-length]
{
  int i = 5;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable name 'i' is too short, expected at least 3 characters [readability-identifier-length]

  int jj = z;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable name 'jj' is too short, expected at least 3 characters [readability-identifier-length]

  for (int m = 0; m < 5; ++m)
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: loop variable name 'm' is too short, expected at least 2 characters [readability-identifier-length]
  {
    doIt();
  }

  try {
    doIt();
  } catch (const myexcept &x)
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: exception variable name 'x' is too short, expected at least 2 characters [readability-identifier-length]
  {
    doIt();
  }
}

void longEnoughVariableNames(int n) // argument 'n' ignored by default configuration
{
  int var = 5;

  for (int i = 0; i < 42; ++i) // 'i' is default allowed, for historical reasons
  {
    doIt();
  }

  for (int kk = 0; kk < 42; ++kk) {
    doIt();
  }

  try {
    doIt();
  } catch (const simpleexcept &e) // ignored by default configuration
  {
    doIt();
  } catch (const myexcept &ex) {
    doIt();
  }

  int x = 5; // ignored by configuration
}
