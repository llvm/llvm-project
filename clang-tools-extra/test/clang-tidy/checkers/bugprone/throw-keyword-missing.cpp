// RUN: %check_clang_tidy %s bugprone-throw-keyword-missing %t -- -- -fexceptions

namespace std {

// std::string declaration (taken from test/clang-tidy/readability-redundant-string-cstr-msvc.cpp).
template <typename T>
class allocator {};
template <typename T>
class char_traits {};
template <typename C, typename T = std::char_traits<C>, typename A = std::allocator<C>>
struct basic_string {
  basic_string();
  basic_string(const basic_string &);
  // MSVC headers define two constructors instead of using optional arguments.
  basic_string(const C *);
  basic_string(const C *, const A &);
  ~basic_string();
};
typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;

// std::exception and std::runtime_error declaration.
struct exception {
  exception();
  exception(const exception &other);
  virtual ~exception();
};

struct runtime_error : public exception {
  explicit runtime_error(const std::string &what_arg);
};

} // namespace std

// The usage of this class should never emit a warning.
struct RegularClass {};

// Class name contains the substring "exception", in certain cases using this class should emit a warning.
struct RegularException {
  RegularException() {}

  // Constructors with a single argument are treated differently (cxxFunctionalCastExpr).
  RegularException(int) {}
};

// --------------

void stdExceptionNotTrownTest(int i) {
  if (i < 0)
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: suspicious exception object created but not thrown; did you mean 'throw {{.*}}'? [bugprone-throw-keyword-missing]
    std::exception();

  if (i > 0)
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: suspicious exception
    std::runtime_error("Unexpected argument");
}

void stdExceptionThrownTest(int i) {
  if (i < 0)
    throw std::exception();

  if (i > 0)
    throw std::runtime_error("Unexpected argument");
}

void regularClassNotThrownTest(int i) {
  if (i < 0)
    RegularClass();
}

void regularClassThrownTest(int i) {
  if (i < 0)
    throw RegularClass();
}

void nameContainsExceptionNotThrownTest(int i) {
  if (i < 0)
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: suspicious exception
    RegularException();

  if (i > 0)
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: suspicious exception
    RegularException(5);
}

void nameContainsExceptionThrownTest(int i) {
  if (i < 0)
    throw RegularException();

  if (i > 0)
    throw RegularException(5);
}

template <class Exception>
void f(int i, Exception excToBeThrown) {}

template <class SomeType>
void templ(int i) {
  if (i > 0)
    SomeType();
}

void funcCallWithTempExcTest() {
  f(5, RegularException());

  templ<RegularException>(4);
  templ<RegularClass>(4);
}

// Global variable initialization test.
RegularException exc = RegularException();
RegularException *excptr = new RegularException();

void localVariableInitTest() {
  RegularException exc = RegularException();
  RegularException *excptr = new RegularException();
}

class CtorInitializerListTest {
  RegularException exc;
  RegularException exc2{};

  CtorInitializerListTest() : exc(RegularException()) {}

  CtorInitializerListTest(int) try : exc(RegularException()) {
    // Constructor body
  } catch (...) {
    // CHECK-MESSAGES: :[[@LINE+1]]:5: warning: suspicious exception
    RegularException();
  }

  CtorInitializerListTest(float);
};

CtorInitializerListTest::CtorInitializerListTest(float) try : exc(RegularException()) {
  // Constructor body
} catch (...) {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: suspicious exception
  RegularException();
}

RegularException funcReturningExceptionTest(int i) {
  return RegularException();
}

void returnedValueTest() {
  funcReturningExceptionTest(3);
}

struct ClassBracedInitListTest {
  ClassBracedInitListTest(RegularException exc) {}
};

void foo(RegularException, ClassBracedInitListTest) {}

void bracedInitListTest() {
  RegularException exc{};
  ClassBracedInitListTest test = {RegularException()};
  foo({}, {RegularException()});
}

typedef std::exception ERROR_BASE;
class RegularError : public ERROR_BASE {};

void typedefTest() {
  // CHECK-MESSAGES: :[[@LINE+1]]:3: warning: suspicious exception
  RegularError();
}

struct ExceptionRAII {
  ExceptionRAII() {}
  ~ExceptionRAII() {}
};

void exceptionRAIITest() {
  ExceptionRAII E;
}

namespace std {
typedef decltype(sizeof(void*)) size_t;
}

void* operator new(std::size_t, void*);

void placeMentNewTest() {
  alignas(RegularException) unsigned char expr[sizeof(RegularException)];
  new (expr) RegularException{};
}
