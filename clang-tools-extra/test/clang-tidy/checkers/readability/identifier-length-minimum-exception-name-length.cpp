// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {readability-identifier-length.MinimumExceptionNameLength: 4}}' \
// RUN: -- -fexceptions

struct myexcept { int val; };
void doIt();

void test() {
  try {
    doIt();
  } catch (const myexcept &err)
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: exception variable name 'err' is too short, expected at least 4 characters [readability-identifier-length]
  {
    doIt();
  }
}
