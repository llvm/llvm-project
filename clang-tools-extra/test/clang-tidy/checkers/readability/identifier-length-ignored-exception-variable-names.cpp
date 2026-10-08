// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {readability-identifier-length.IgnoredExceptionVariableNames: "^[ex]$"}}' \
// RUN: -- -fexceptions

struct myexcept { int val; };
void doIt();

void test() {
  try {
    doIt();
  } catch (const myexcept &e) { doIt(); } // no warning, e allowed
  try {
    doIt();
  } catch (const myexcept &x) { doIt(); } // no warning, x allowed
  try {
    doIt();
  } catch (const myexcept &y)
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: exception variable name 'y' is too short, expected at least 2 characters [readability-identifier-length]
  {
    doIt();
  }
}
