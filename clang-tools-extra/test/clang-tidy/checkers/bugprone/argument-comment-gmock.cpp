// RUN: %check_clang_tidy %s bugprone-argument-comment %t

namespace testing {
namespace internal {

template <typename F>
struct Function;

template <typename R>
struct Function<R()> {
  typedef R Result;
};

template <typename R, typename A1>
struct Function<R(A1)>
    : Function<R()> {
  typedef A1 Argument1;
};

template <typename R, typename A1, typename A2>
struct Function<R(A1, A2)>
    : Function<R(A1)> {
  typedef A2 Argument2;
};

} // namespace internal

template <typename F>
class MockSpec {
 public:
  void f();
};

template <typename T>
class Matcher {
 public:
  explicit Matcher();
  Matcher(T value);
};

} // namespace testing

#define GMOCK_RESULT_(tn, ...) \
    tn ::testing::internal::Function<__VA_ARGS__>::Result
#define GMOCK_ARG_(tn, N, ...) \
    tn ::testing::internal::Function<__VA_ARGS__>::Argument##N
#define GMOCK_MATCHER_(tn, N, ...) \
    const ::testing::Matcher<GMOCK_ARG_(tn, N, __VA_ARGS__)>&
#define GMOCK_METHOD2_(tn, constness, ct, Method, ...)            \
  GMOCK_RESULT_(tn, __VA_ARGS__)                                  \
  ct Method(                                                      \
      GMOCK_ARG_(tn, 1, __VA_ARGS__) gmock_a1,                    \
      GMOCK_ARG_(tn, 2, __VA_ARGS__) gmock_a2) constness;         \
  ::testing::MockSpec<__VA_ARGS__>                                \
      gmock_##Method(GMOCK_MATCHER_(tn, 1, __VA_ARGS__) gmock_a1, \
                     GMOCK_MATCHER_(tn, 2, __VA_ARGS__) gmock_a2) constness
#define MOCK_METHOD2(m, ...) GMOCK_METHOD2_(, , , m, __VA_ARGS__)
#define MOCK_CONST_METHOD2(m, ...) GMOCK_METHOD2_(, const, , m, __VA_ARGS__)
#define GMOCK_EXPECT_CALL_IMPL_(obj, call) \
    ((obj).gmock_##call).f()
#define EXPECT_CALL(obj, call) GMOCK_EXPECT_CALL_IMPL_(obj, call)

class Base {
 public:
  virtual void Method(int param_one_base, int param_two_base);
};
class Derived : public Base {
 public:
  virtual void Method(int param_one, int param_two);
  virtual void Method2(int p_one, int p_two) const;
};
class MockDerived : public Derived {
 public:
  MOCK_METHOD2(Method, void(int a, int b));
  MOCK_CONST_METHOD2(Method2, void(int c, int d));
};

class MockStandalone {
 public:
  MOCK_METHOD2(Method, void(int aaa, int bbb));
};

void test_gmock_expectations() {
  MockDerived m;
  EXPECT_CALL(m, Method(/*param_one=*/1, /*param_tw=*/2));
// CHECK-NOTES: [[@LINE-1]]:42: warning: argument name 'param_tw' in comment does not match parameter name 'param_two'
// CHECK-NOTES: [[@LINE-18]]:42: note: 'param_two' declared here
// CHECK-NOTES: [[@LINE-14]]:3: note: actual callee ('gmock_Method') is declared here
// CHECK-NOTES: [[@LINE-32]]:30: note: expanded from macro 'MOCK_METHOD2'
// CHECK-NOTES: [[@LINE-35]]:7: note: expanded from macro 'GMOCK_METHOD2_'
// CHECK-NOTES: note: expanded from here
// CHECK-FIXES:   EXPECT_CALL(m, Method(/*param_one=*/1, /*param_two=*/2));
  EXPECT_CALL(m, Method2(/*p_on=*/3, /*p_two=*/4));
// CHECK-NOTES: [[@LINE-1]]:26: warning: argument name 'p_on' in comment does not match parameter name 'p_one'
// CHECK-NOTES: [[@LINE-25]]:28: note: 'p_one' declared here
// CHECK-NOTES: [[@LINE-21]]:3: note: actual callee ('gmock_Method2') is declared here
// CHECK-NOTES: [[@LINE-39]]:36: note: expanded from macro 'MOCK_CONST_METHOD2'
// CHECK-NOTES: [[@LINE-43]]:7: note: expanded from macro 'GMOCK_METHOD2_'
// CHECK-NOTES: note: expanded from here
// CHECK-FIXES:   EXPECT_CALL(m, Method2(/*p_one=*/3, /*p_two=*/4));

  #define PARAM1 11
  #define PARAM2 22
  EXPECT_CALL(m, Method2(/*p_on1=*/PARAM1, /*p_tw2=*/PARAM2));
// CHECK-NOTES: [[@LINE-1]]:26: warning: argument name 'p_on1' in comment does not match parameter name 'p_one'
// CHECK-NOTES: [[@LINE-36]]:28: note: 'p_one' declared here
// CHECK-NOTES: [[@LINE-32]]:3: note: actual callee ('gmock_Method2') is declared here
// CHECK-NOTES: [[@LINE-50]]:36: note: expanded from macro 'MOCK_CONST_METHOD2'
// CHECK-NOTES: [[@LINE-54]]:7: note: expanded from macro 'GMOCK_METHOD2_'
// CHECK-NOTES: note: expanded from here
// CHECK-NOTES: [[@LINE-7]]:44: warning: argument name 'p_tw2' in comment does not match parameter name 'p_two'
// CHECK-NOTES: [[@LINE-42]]:39: note: 'p_two' declared here
// CHECK-NOTES: [[@LINE-38]]:3: note: actual callee ('gmock_Method2') is declared here
// CHECK-NOTES: [[@LINE-56]]:36: note: expanded from macro 'MOCK_CONST_METHOD2'
// CHECK-NOTES: [[@LINE-60]]:7: note: expanded from macro 'GMOCK_METHOD2_'
// CHECK-NOTES: note: expanded from here
// CHECK-FIXES:   EXPECT_CALL(m, Method2(/*p_one=*/PARAM1, /*p_two=*/PARAM2));

  MockStandalone m2;
  EXPECT_CALL(m2, Method(/*aaa=*/5, /*bbc=*/6));
}

void test_gmock_direct_calls() {
  MockDerived m;
  m.Method(/*param_one=*/1, /*param_tw=*/2);
// CHECK-NOTES: [[@LINE-1]]:29: warning: argument name 'param_tw' in comment does not match parameter name 'param_two'
// CHECK-NOTES: [[@LINE-58]]:42: note: 'param_two' declared here
// CHECK-NOTES: [[@LINE-54]]:16: note: actual callee ('Method') is declared here
// CHECK-FIXES:   m.Method(/*param_one=*/1, /*param_two=*/2);
}
