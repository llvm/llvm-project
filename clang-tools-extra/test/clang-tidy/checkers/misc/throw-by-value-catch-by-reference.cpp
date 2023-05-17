// RUN: %check_clang_tidy %s misc-throw-by-value-catch-by-reference %t -- -- -fcxx-exceptions


class logic_error {
public:
  logic_error(const char *message) {}
};

typedef logic_error *logic_ptr;
typedef logic_ptr logic_double_typedef;

int lastException;

template <class T> struct remove_reference { typedef T type; };
template <class T> struct remove_reference<T &> { typedef T type; };
template <class T> struct remove_reference<T &&> { typedef T type; };

template <typename T> typename remove_reference<T>::type &&move(T &&arg) {
  return static_cast<typename remove_reference<T>::type &&>(arg);
}

logic_error CreateException() { return logic_error("created"); }

void testThrowFunc() {
  throw new logic_error("by pointer");
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: throw expression throws a pointer; it should throw a non-pointer value instead [misc-throw-by-value-catch-by-reference]
  logic_ptr tmp = new logic_error("by pointer");
  throw tmp;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: throw expression should throw anonymous temporary values instead [misc-throw-by-value-catch-by-reference]
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: throw expression throws a pointer; it should throw a non-pointer value instead [misc-throw-by-value-catch-by-reference]
  throw logic_error("by value");
  auto *literal = "test";
  throw logic_error(literal);
  throw "test string literal";
  throw L"throw wide string literal";
  const char *characters = 0;
  throw characters;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: throw expression should throw anonymous temporary values instead [misc-throw-by-value-catch-by-reference]
  // CHECK-MESSAGES: :[[@LINE-2]]:9: warning: throw expression throws a pointer; it should throw a non-pointer value instead [misc-throw-by-value-catch-by-reference]
  logic_error lvalue("lvalue");
  throw lvalue;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: throw expression should throw anonymous temporary values instead [misc-throw-by-value-catch-by-reference]

  throw move(lvalue);
  int &ex = lastException;
  throw ex;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: throw expression should throw anonymous temporary values instead [misc-throw-by-value-catch-by-reference]
  throw CreateException();
}

void throwReferenceFunc(logic_error &ref) { throw ref; }

void catchByPointer() {
  try {
    testThrowFunc();
  } catch (logic_error *e) {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: catch handler catches a pointer value; should throw a non-pointer value and catch by reference instead [misc-throw-by-value-catch-by-reference]
  }
}

void catchByValue() {
  try {
    testThrowFunc();
  } catch (logic_error e) {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: catch handler catches by value; should catch by reference instead [misc-throw-by-value-catch-by-reference]
  }
}

void catchByReference() {
  try {
    testThrowFunc();
  } catch (logic_error &e) {
  }
}

void catchByConstReference() {
  try {
    testThrowFunc();
  } catch (const logic_error &e) {
  }
}

void catchTypedef() {
  try {
    testThrowFunc();
  } catch (logic_ptr) {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: catch handler catches a pointer value; should throw a non-pointer value and catch by reference instead [misc-throw-by-value-catch-by-reference]
  }
}

void catchAll() {
  try {
    testThrowFunc();
  } catch (...) {
  }
}

void catchLiteral() {
  try {
    testThrowFunc();
  } catch (const char *) {
  } catch (const wchar_t *) {
    // disabled for now until it is clear
    // how to enable them in the test
    //} catch (const char16_t*) {
    //} catch (const char32_t*) {
  }
}

// catching fundamentals should not warn
void catchFundamental() {
  try {
    testThrowFunc();
  } catch (int) {
  } catch (double) {
  } catch (unsigned long) {
  }
}

struct TrivialType {
  double x;
  double y;
};

void catchTrivial() {
  try {
    testThrowFunc();
  } catch (TrivialType) {
  }
}

typedef logic_error &fine;
void additionalTests() {
  try {
  } catch (int i) {  // ok
    throw i;         // ok
  } catch (fine e) { // ok
    throw e;         // ok
  } catch (logic_error *e) {
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: catch handler catches a pointer value; should throw a non-pointer value and catch by reference instead [misc-throw-by-value-catch-by-reference]
    throw e;      // ok, despite throwing a pointer
  } catch (...) { // ok
    throw;        // ok
  }
}

struct S {};

S &returnByReference();
S returnByValue();

void f() {
  throw returnByReference(); // Should diagnose
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: throw expression should throw anonymous temporary values instead [misc-throw-by-value-catch-by-reference]
  throw returnByValue(); // Should not diagnose
}
