// RUN: %check_clang_tidy %s modernize-use-uncaught-exceptions %t -- -- -std=c++1z
#define MACRO std::uncaught_exception
// CHECK-FIXES: #define MACRO std::uncaught_exception

bool uncaught_exception() {
  return 0;
}

namespace std {
  bool uncaught_exception() {
    return false;
  }

  int uncaught_exceptions() {
    return 0;
  }
}

template <typename T>
bool doSomething(T t) { 
  return t();
  // CHECK-FIXES: return t();
}

template <bool (*T)()>
bool doSomething2() { 
  return T();
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: return T();
}

void no_warn() {

  uncaught_exception();
  // CHECK-FIXES: uncaught_exception();

  doSomething(uncaught_exception);
  // CHECK-FIXES: doSomething(uncaught_exception);
}

void warn() {

  std::uncaught_exception();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: std::uncaught_exceptions();

  using std::uncaught_exception;
  // CHECK-MESSAGES: [[@LINE-1]]:14: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: using std::uncaught_exceptions;

  uncaught_exception();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: uncaught_exceptions();

  bool b{uncaught_exception()};
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: bool b{std::uncaught_exceptions() > 0};

  MACRO();
  // CHECK-MESSAGES: [[@LINE-1]]:3: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: MACRO();

  doSomething(std::uncaught_exception);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: doSomething(std::uncaught_exception);

  doSomething(uncaught_exception);
  // CHECK-MESSAGES: [[@LINE-1]]:15: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: doSomething(uncaught_exception);

  bool (*foo)();
  foo = &uncaught_exception;
  // CHECK-MESSAGES: [[@LINE-1]]:10: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: foo = &uncaught_exception;

  doSomething2<uncaught_exception>();
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: 'std::uncaught_exception' is deprecated, use 'std::uncaught_exceptions' instead
  // CHECK-FIXES: doSomething2<uncaught_exception>();
}
