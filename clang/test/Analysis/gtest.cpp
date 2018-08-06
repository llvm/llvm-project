//RUN: %clang_analyze_cc1 -cc1 -std=c++11  -analyzer-checker=core,apiModeling.google.GTest,debug.ExprInspection -analyzer-eagerly-assume %s -verify
//RUN: %clang_analyze_cc1 -cc1 -std=c++11  -analyzer-checker=core,apiModeling.google.GTest,debug.ExprInspection -analyzer-eagerly-assume -DGTEST_VERSION_1_8_AND_LATER=1 %s -verify

void clang_analyzer_eval(int);
void clang_analyzer_warnIfReached();

namespace std {
  class string {
    public:
    ~string();
    const char *c_str();
  };
}

namespace testing {

class Message { };
class TestPartResult {
 public:
  enum Type {
    kSuccess,
    kNonFatalFailure,
    kFatalFailure
  };
};

namespace internal {

class AssertHelper {
 public:
  AssertHelper(TestPartResult::Type type, const char* file, int line,
               const char* message);
  ~AssertHelper();
  void operator=(const Message& message) const;
};


template <typename T>
struct AddReference { typedef T& type; };
template <typename T>
struct AddReference<T&> { typedef T& type; };
template <typename From, typename To>
class ImplicitlyConvertible {
 private:
  static typename AddReference<From>::type MakeFrom();
  static char Helper(To);
  static char (&Helper(...))[2];
 public:
  static const bool value =
      sizeof(Helper(ImplicitlyConvertible::MakeFrom())) == 1;
};
template <typename From, typename To>
const bool ImplicitlyConvertible<From, To>::value;
template<bool> struct EnableIf;
template<> struct EnableIf<true> { typedef void type; };

} // end internal


class AssertionResult {
public:

  // The implementation for the copy constructor is not exposed in the
  // interface.
  AssertionResult(const AssertionResult& other);

#if defined(GTEST_VERSION_1_8_AND_LATER)
  template <typename T>
  explicit AssertionResult(
      const T& success,
      typename internal::EnableIf<
          !internal::ImplicitlyConvertible<T, AssertionResult>::value>::type*
          /*enabler*/ = 0)
      : success_(success) {}
#else
  explicit AssertionResult(bool success) : success_(success) {}
#endif

  operator bool() const { return success_; }

  // The actual AssertionResult does not have an explicit destructor, but
  // it does have a non-trivial member veriable, so we add a destructor here
  // to force temporary cleanups.
  ~AssertionResult();
private:

  bool success_;
};

namespace internal {
std::string GetBoolAssertionFailureMessage(
    const AssertionResult& assertion_result,
    const char* expression_text,
    const char* actual_predicate_value,
    const char* expected_predicate_value);
} // end internal

} // end testing

#define GTEST_MESSAGE_AT_(file, line, message, result_type) \
  ::testing::internal::AssertHelper(result_type, file, line, message) \
    = ::testing::Message()

#define GTEST_MESSAGE_(message, result_type) \
  GTEST_MESSAGE_AT_(__FILE__, __LINE__, message, result_type)

#define GTEST_FATAL_FAILURE_(message) \
  return GTEST_MESSAGE_(message, ::testing::TestPartResult::kFatalFailure)

#define GTEST_NONFATAL_FAILURE_(message) \
  GTEST_MESSAGE_(message, ::testing::TestPartResult::kNonFatalFailure)

# define GTEST_AMBIGUOUS_ELSE_BLOCKER_ switch (0) case 0: default:

#define GTEST_TEST_BOOLEAN_(expression, text, actual, expected, fail) \
  GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
  if (const ::testing::AssertionResult gtest_ar_ = \
      ::testing::AssertionResult(expression)) \
    ; \
  else \
    fail(::testing::internal::GetBoolAssertionFailureMessage(\
        gtest_ar_, text, #actual, #expected).c_str())

#define EXPECT_TRUE(condition) \
  GTEST_TEST_BOOLEAN_((condition), #condition, false, true, \
                      GTEST_NONFATAL_FAILURE_)
#define ASSERT_TRUE(condition) \
  GTEST_TEST_BOOLEAN_((condition), #condition, false, true, \
                      GTEST_FATAL_FAILURE_)

#define ASSERT_FALSE(condition) \
  GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
                      GTEST_FATAL_FAILURE_)

void testAssertTrue(int *p) {
  ASSERT_TRUE(p != nullptr);
  EXPECT_TRUE(1 == *p); // no-warning
}

void testAssertFalse(int *p) {
  ASSERT_FALSE(p == nullptr);
  EXPECT_TRUE(1 == *p); // no-warning
}

void testConstrainState(int p) {
  ASSERT_TRUE(p == 7);

  clang_analyzer_eval(p == 7); // expected-warning {{TRUE}}

  ASSERT_TRUE(false);
  clang_analyzer_warnIfReached(); // no-warning
}

void testAssertSymbolicPtr(const bool *b) {
  ASSERT_TRUE(*b); // no-crash

  clang_analyzer_eval(*b); // expected-warning{{TRUE}}
}

void testAssertSymbolicRef(const bool &b) {
  ASSERT_TRUE(b); // no-crash

  clang_analyzer_eval(b); // expected-warning{{TRUE}}
}
