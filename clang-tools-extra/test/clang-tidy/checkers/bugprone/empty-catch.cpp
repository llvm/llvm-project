// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-empty-catch %t -- \
// RUN: -config="{CheckOptions: {bugprone-empty-catch.AllowEmptyCatchForExceptions: '::SafeException;WarnException', \
// RUN:        bugprone-empty-catch.IgnoreCatchWithKeywords: '@IGNORE;@TODO'}}" -- -fexceptions

struct Exception {};
struct SafeException {};
struct WarnException : Exception {};

int functionWithThrow() {
  try {
    throw 5;
  } catch (const Exception &) {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: empty catch statements hide issues; to handle exceptions appropriately, consider re-throwing, handling, or avoiding catch altogether [bugprone-empty-catch]
  } catch (...) {
  // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: empty catch statements hide issues; to handle exceptions appropriately, consider re-throwing, handling, or avoiding catch altogether [bugprone-empty-catch]
  }
  return 0;
}

int functionWithHandling() {
  try {
    throw 5;
  } catch (const Exception &) {
    return 2;
  } catch (...) {
    return 1;
  }
  return 0;
}

int functionWithReThrow() {
  try {
    throw 5;
  } catch (...) {
    throw;
  }
}

int functionWithNewThrow() {
  try {
    throw 5;
  } catch (...) {
    throw Exception();
  }
}

void functionWithAllowedException() {
  try {

  } catch (const SafeException &) {
  } catch (WarnException) {
  }
}

void functionWithComment() {
  try {
  } catch (const Exception &) {
    // @todo: implement later, check case insensitive
  }
}

void functionWithComment2() {
  try {
  } catch (const Exception &) {
    // @IGNORE: relax its safe
  }
}
