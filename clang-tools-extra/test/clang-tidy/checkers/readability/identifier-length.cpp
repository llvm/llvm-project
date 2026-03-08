// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: \
// RUN:  {readability-identifier-length.IgnoredVariableNames: "^[xy]$", readability-identifier-length.LineCountThreshold: 1}}' \
// RUN: -- -fexceptions

struct myexcept {
  int val;
};

struct simpleexcept {
  int other;
};

template<typename... Ts>
void doIt(Ts...);

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
    doIt(i, jj, m);
  }

  try {
    doIt(z);
  } catch (const myexcept &x)
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: exception variable name 'x' is too short, expected at least 2 characters [readability-identifier-length]
  {
    doIt(x);
  }
}

void longEnoughVariableNames(int n, int m) // argument 'n' ignored by default configuration, 'm' is only used on this line
{
  int var = 5;

  for (int i = 0; i < 42; ++i) // 'i' is default allowed, for historical reasons
  {
    doIt(var, i);
  }

  for (int a = 0; a < 42; ++a) // 'a' is only used on this line
  {
      doIt();
  }

  for (int kk = 0; kk < 42; ++kk) {
    doIt(kk);
  }

  try {
    doIt(n);
  } catch (const simpleexcept &e) // ignored by default configuration
  {
    doIt(e);
  } catch (const myexcept &ex) {
    doIt(ex);
  } catch (const int &d) { doIt(d); } // 'd' is only used on this line

  int x = 5; // ignored by configuration
  ++x;

  int b = 0; // 'b' is only used on this line
}
