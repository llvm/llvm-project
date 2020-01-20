//===------------------ Base class for libc unittests -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This file should stricly not include any other file. Not even standard
// library headers.

namespace llvm_libc {
namespace testing {

// We define our own EnableIf and IsIntegerType traits because we do not want to
// include even the standard header <type_traits>.
template <bool B, typename T> struct EnableIf;
template <typename T> struct EnableIf<true, T> { typedef T Type; };

template <bool B, typename T>
using EnableIfType = typename EnableIf<B, T>::Type;

template <typename Type> struct IsIntegerType {
  static const bool Value = false;
};

template <> struct IsIntegerType<char> { static const bool Value = true; };
template <> struct IsIntegerType<unsigned char> {
  static const bool Value = true;
};

template <> struct IsIntegerType<short> { static const bool Value = true; };
template <> struct IsIntegerType<unsigned short> {
  static const bool Value = true;
};

template <> struct IsIntegerType<int> { static const bool Value = true; };
template <> struct IsIntegerType<unsigned int> {
  static const bool Value = true;
};

template <> struct IsIntegerType<long> { static const bool Value = true; };
template <> struct IsIntegerType<unsigned long> {
  static const bool Value = true;
};

template <> struct IsIntegerType<long long> { static const bool Value = true; };
template <> struct IsIntegerType<unsigned long long> {
  static const bool Value = true;
};

template <typename T> struct IsPointerType;

template <typename T> struct IsPointerType<T *> {
  static const bool Value = true;
};

class RunContext;

// Only the following conditions are supported. Notice that we do not have
// a TRUE or FALSE condition. That is because, C library funtions do not
// return, but use integral return values to indicate true or false
// conditions. Hence, it is more appropriate to use the other comparison
// condtions for such cases.
enum TestCondition {
  Cond_None,
  Cond_EQ,
  Cond_NE,
  Cond_LT,
  Cond_LE,
  Cond_GT,
  Cond_GE,
};

namespace internal {

template <typename ValType>
bool test(RunContext &Ctx, TestCondition Cond, ValType LHS, ValType RHS,
          const char *LHSStr, const char *RHSStr, const char *File,
          unsigned long Line);

} // namespace internal

// NOTE: One should not create instances and call methods on them directly. One
// should use the macros TEST or TEST_F to write test cases.
class Test {
private:
  Test *Next = nullptr;

public:
  virtual ~Test() {}
  virtual void SetUp() {}
  virtual void TearDown() {}

  static int runTests();

protected:
  static void addTest(Test *T);

  // We make use of a template function, with |LHS| and |RHS| as explicit
  // parameters, for enhanced type checking. Other gtest like test unittest
  // frameworks have a similar functions which takes a boolean argument
  // instead of the explicit |LHS| and |RHS| arguments. This boolean argument
  // is the result of the |Cond| operation on |LHS| and |RHS|. Though not bad,
  // mismatched |LHS| and |RHS| types can potentially succeed because of type
  // promotion.
  template <typename ValType,
            EnableIfType<IsIntegerType<ValType>::Value, ValType> = 0>
  static bool test(RunContext &Ctx, TestCondition Cond, ValType LHS,
                   ValType RHS, const char *LHSStr, const char *RHSStr,
                   const char *File, unsigned long Line) {
    return internal::test(Ctx, Cond, LHS, RHS, LHSStr, RHSStr, File, Line);
  }

  template <typename ValType,
            EnableIfType<IsPointerType<ValType>::Value, ValType> = nullptr>
  static bool test(RunContext &Ctx, TestCondition Cond, ValType LHS,
                   ValType RHS, const char *LHSStr, const char *RHSStr,
                   const char *File, unsigned long Line) {
    return internal::test(Ctx, Cond, (unsigned long long)LHS,
                          (unsigned long long)RHS, LHSStr, RHSStr, File, Line);
  }

  static bool testStrEq(RunContext &Ctx, const char *LHS, const char *RHS,
                        const char *LHSStr, const char *RHSStr,
                        const char *File, unsigned long Line);

  static bool testStrNe(RunContext &Ctx, const char *LHS, const char *RHS,
                        const char *LHSStr, const char *RHSStr,
                        const char *File, unsigned long Line);

private:
  virtual void Run(RunContext &Ctx) = 0;
  virtual const char *getName() const = 0;

  static Test *Start;
  static Test *End;
};

} // namespace testing
} // namespace llvm_libc

#define TEST(SuiteName, TestName)                                              \
  class SuiteName##_##TestName : public llvm_libc::testing::Test {             \
  public:                                                                      \
    SuiteName##_##TestName() { addTest(this); }                                \
    void Run(llvm_libc::testing::RunContext &) override;                       \
    const char *getName() const override { return #SuiteName "." #TestName; }  \
  };                                                                           \
  SuiteName##_##TestName SuiteName##_##TestName##_Instance;                    \
  void SuiteName##_##TestName::Run(llvm_libc::testing::RunContext &Ctx)

#define TEST_F(SuiteClass, TestName)                                           \
  class SuiteClass##_##TestName : public SuiteClass {                          \
  public:                                                                      \
    SuiteClass##_##TestName() { addTest(this); }                               \
    void Run(llvm_libc::testing::RunContext &) override;                       \
    const char *getName() const override { return #SuiteClass "." #TestName; } \
  };                                                                           \
  SuiteClass##_##TestName SuiteClass##_##TestName##_Instance;                  \
  void SuiteClass##_##TestName::Run(llvm_libc::testing::RunContext &Ctx)

#define EXPECT_EQ(LHS, RHS)                                                    \
  llvm_libc::testing::Test::test(Ctx, llvm_libc::testing::Cond_EQ, (LHS),      \
                                 (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_EQ(LHS, RHS)                                                    \
  if (!EXPECT_EQ(LHS, RHS))                                                    \
  return

#define EXPECT_NE(LHS, RHS)                                                    \
  llvm_libc::testing::Test::test(Ctx, llvm_libc::testing::Cond_NE, (LHS),      \
                                 (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_NE(LHS, RHS)                                                    \
  if (!EXPECT_NE(LHS, RHS))                                                    \
  return

#define EXPECT_LT(LHS, RHS)                                                    \
  llvm_libc::testing::Test::test(Ctx, llvm_libc::testing::Cond_LT, (LHS),      \
                                 (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_LT(LHS, RHS)                                                    \
  if (!EXPECT_LT(LHS, RHS))                                                    \
  return

#define EXPECT_LE(LHS, RHS)                                                    \
  llvm_libc::testing::Test::test(Ctx, llvm_libc::testing::Cond_LE, (LHS),      \
                                 (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_LE(LHS, RHS)                                                    \
  if (!EXPECT_LE(LHS, RHS))                                                    \
  return

#define EXPECT_GT(LHS, RHS)                                                    \
  llvm_libc::testing::Test::test(Ctx, llvm_libc::testing::Cond_GT, (LHS),      \
                                 (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_GT(LHS, RHS)                                                    \
  if (!EXPECT_GT(LHS, RHS))                                                    \
  return

#define EXPECT_GE(LHS, RHS)                                                    \
  llvm_libc::testing::Test::test(Ctx, llvm_libc::testing::Cond_GE, (LHS),      \
                                 (RHS), #LHS, #RHS, __FILE__, __LINE__)
#define ASSERT_GE(LHS, RHS)                                                    \
  if (!EXPECT_GE(LHS, RHS))                                                    \
  return

#define EXPECT_STREQ(LHS, RHS)                                                 \
  llvm_libc::testing::Test::testStrEq(Ctx, (LHS), (RHS), #LHS, #RHS, __FILE__, \
                                      __LINE__)
#define ASSERT_STREQ(LHS, RHS)                                                 \
  if (!EXPECT_STREQ(LHS, RHS))                                                 \
  return

#define EXPECT_STRNE(LHS, RHS)                                                 \
  llvm_libc::testing::Test::testStrNe(Ctx, (LHS), (RHS), #LHS, #RHS, __FILE__, \
                                      __LINE__)
#define ASSERT_STRNE(LHS, RHS)                                                 \
  if (!EXPECT_STRNE(LHS, RHS))                                                 \
  return
