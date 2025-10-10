//===- UncheckedStatusOrAccessModelTestFixture.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "UncheckedStatusOrAccessModelTestFixture.h"
#include "llvm/Support/ErrorHandling.h"

#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"

namespace clang::dataflow::statusor_model {
namespace {

static constexpr const char *kAbslDefsHeader = R"cc(
  // Contains minimal mock declarations of common entities from //base, //util
  // and //testing.

#pragma clang system_header

#ifndef BASE_DEFS_H_
#define BASE_DEFS_H_

#include "absl_type_traits.h"
#include "std_type_traits.h"
#include "stdlib_defs.h"
#include "string_defs.h"

  namespace absl {
  struct in_place_t {};

  constexpr in_place_t in_place;

  struct nullopt_t {
    constexpr explicit nullopt_t() {}
  };

  constexpr nullopt_t nullopt;
  }  // namespace absl

  namespace std {

  struct in_place_t {};

  constexpr in_place_t in_place;

  struct nullopt_t {
    constexpr explicit nullopt_t() {}
  };

  constexpr nullopt_t nullopt;

  template <class _Tp>
  struct __optional_destruct_base {
    constexpr void reset() noexcept;
  };

  template <class _Tp>
  struct __optional_storage_base : __optional_destruct_base<_Tp> {
    constexpr bool has_value() const noexcept;
  };

  template <typename _Tp>
  class optional : private __optional_storage_base<_Tp> {
    using __base = __optional_storage_base<_Tp>;

   public:
    using value_type = _Tp;

   private:
    struct _CheckOptionalArgsConstructor {
      template <class _Up>
      static constexpr bool __enable_implicit() {
        return is_constructible_v<_Tp, _Up&&> && is_convertible_v<_Up&&, _Tp>;
      }

      template <class _Up>
      static constexpr bool __enable_explicit() {
        return is_constructible_v<_Tp, _Up&&> && !is_convertible_v<_Up&&, _Tp>;
      }
    };
    template <class _Up>
    using _CheckOptionalArgsCtor =
        _If<_IsNotSame<__uncvref_t<_Up>, in_place_t>::value &&
                _IsNotSame<__uncvref_t<_Up>, optional>::value,
            _CheckOptionalArgsConstructor, __check_tuple_constructor_fail>;
    template <class _QualUp>
    struct _CheckOptionalLikeConstructor {
      template <class _Up, class _Opt = optional<_Up>>
      using __check_constructible_from_opt =
          _Or<is_constructible<_Tp, _Opt&>, is_constructible<_Tp, _Opt const&>,
              is_constructible<_Tp, _Opt&&>,
              is_constructible<_Tp, _Opt const&&>, is_convertible<_Opt&, _Tp>,
              is_convertible<_Opt const&, _Tp>, is_convertible<_Opt&&, _Tp>,
              is_convertible<_Opt const&&, _Tp>>;
      template <class _Up, class _QUp = _QualUp>
      static constexpr bool __enable_implicit() {
        return is_convertible<_QUp, _Tp>::value &&
               !__check_constructible_from_opt<_Up>::value;
      }
      template <class _Up, class _QUp = _QualUp>
      static constexpr bool __enable_explicit() {
        return !is_convertible<_QUp, _Tp>::value &&
               !__check_constructible_from_opt<_Up>::value;
      }
    };

    template <class _Up, class _QualUp>
    using _CheckOptionalLikeCtor =
        _If<_And<_IsNotSame<_Up, _Tp>, is_constructible<_Tp, _QualUp>>::value,
            _CheckOptionalLikeConstructor<_QualUp>,
            __check_tuple_constructor_fail>;
    template <class _Up, class _QualUp>
    using _CheckOptionalLikeAssign =
        _If<_And<_IsNotSame<_Up, _Tp>, is_constructible<_Tp, _QualUp>,
                 is_assignable<_Tp&, _QualUp>>::value,
            _CheckOptionalLikeConstructor<_QualUp>,
            __check_tuple_constructor_fail>;

   public:
    constexpr optional() noexcept {}
    constexpr optional(const optional&) = default;
    constexpr optional(optional&&) = default;
    constexpr optional(nullopt_t) noexcept {}

    template <class _InPlaceT, class... _Args,
              class = std::enable_if_t<
                  _And<_IsSame<_InPlaceT, in_place_t>,
                       is_constructible<value_type, _Args...>>::value>>
    constexpr explicit optional(_InPlaceT, _Args&&... __args);

    template <class _Up, class... _Args,
              class = std::enable_if_t<is_constructible_v<
                  value_type, initializer_list<_Up>&, _Args...>>>
    constexpr explicit optional(in_place_t, initializer_list<_Up> __il,
                                _Args&&... __args);

    template <class _Up = value_type,
              std::enable_if_t<_CheckOptionalArgsCtor<
                                   _Up>::template __enable_implicit<_Up>(),
                               int> = 0>
    constexpr optional(_Up&& __v);

    template <class _Up, std::enable_if_t<_CheckOptionalArgsCtor<_Up>::
                                              template __enable_explicit<_Up>(),
                                          int> = 0>
    constexpr explicit optional(_Up&& __v);

    template <class _Up,
              std::enable_if_t<_CheckOptionalLikeCtor<_Up, _Up const&>::
                                   template __enable_implicit<_Up>(),
                               int> = 0>
    constexpr optional(const optional<_Up>& __v);

    template <class _Up,
              std::enable_if_t<_CheckOptionalLikeCtor<_Up, _Up const&>::
                                   template __enable_explicit<_Up>(),
                               int> = 0>
    constexpr explicit optional(const optional<_Up>& __v);

    template <class _Up, std::enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::
                                              template __enable_implicit<_Up>(),
                                          int> = 0>
    constexpr optional(optional<_Up>&& __v);

    template <class _Up, std::enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::
                                              template __enable_explicit<_Up>(),
                                          int> = 0>
    constexpr explicit optional(optional<_Up>&& __v);

    constexpr optional& operator=(nullopt_t) noexcept;

    optional& operator=(const optional&);

    optional& operator=(optional&&);

    template <class _Up = value_type,
              class = std::enable_if_t<
                  _And<_IsNotSame<__uncvref_t<_Up>, optional>,
                       _Or<_IsNotSame<__uncvref_t<_Up>, value_type>,
                           _Not<is_scalar<value_type>>>,
                       is_constructible<value_type, _Up>,
                       is_assignable<value_type&, _Up>>::value>>
    constexpr optional& operator=(_Up&& __v);

    template <class _Up,
              std::enable_if_t<_CheckOptionalLikeAssign<_Up, _Up const&>::
                                   template __enable_assign<_Up>(),
                               int> = 0>
    constexpr optional& operator=(const optional<_Up>& __v);

    template <class _Up, std::enable_if_t<_CheckOptionalLikeCtor<_Up, _Up&&>::
                                              template __enable_assign<_Up>(),
                                          int> = 0>
    constexpr optional& operator=(optional<_Up>&& __v);

    const _Tp& operator*() const&;
    _Tp& operator*() &;
    const _Tp&& operator*() const&&;
    _Tp&& operator*() &&;

    const _Tp* operator->() const;
    _Tp* operator->();

    const _Tp& value() const&;
    _Tp& value() &;
    const _Tp&& value() const&&;
    _Tp&& value() &&;

    template <typename U>
    constexpr _Tp value_or(U&& v) const&;
    template <typename U>
    _Tp value_or(U&& v) &&;

    constexpr explicit operator bool() const noexcept;

    using __base::has_value;

    template <typename... Args>
    _Tp& emplace(Args&&... args);

    template <typename U, typename... Args>
    _Tp& emplace(std::initializer_list<U> ilist, Args&&... args);

    using __base::reset;

    constexpr void swap(optional& __opt) noexcept;
  };

  template <class _Tp, class _Up>
  constexpr std::enable_if_t<
      is_convertible_v<decltype(declval<const _Tp&>() == declval<const _Up&>()),
                       bool>,
      bool>
  operator==(const optional<_Tp>& __x, const optional<_Up>& __y);

  template <class _Tp, class _Up>
  constexpr std::enable_if_t<
      is_convertible_v<decltype(declval<const _Tp&>() != declval<const _Up&>()),
                       bool>,
      bool>
  operator!=(const optional<_Tp>& __x, const optional<_Up>& __y);

  template <class _Tp>
  constexpr bool operator==(const optional<_Tp>& __x, nullopt_t) noexcept;

  template <class _Tp>
  constexpr bool operator==(nullopt_t, const optional<_Tp>& __x) noexcept;

  template <class _Tp>
  constexpr bool operator!=(const optional<_Tp>& __x, nullopt_t) noexcept;

  template <class _Tp>
  constexpr bool operator!=(nullopt_t, const optional<_Tp>& __x) noexcept;

  template <class _Tp, class _Up>
  constexpr std::enable_if_t<
      is_convertible_v<decltype(declval<const _Tp&>() == declval<const _Up&>()),
                       bool>,
      bool>
  operator==(const optional<_Tp>& __x, const _Up& __v);

  template <class _Tp, class _Up>
  constexpr std::enable_if_t<
      is_convertible_v<decltype(declval<const _Tp&>() == declval<const _Up&>()),
                       bool>,
      bool>
  operator==(const _Tp& __v, const optional<_Up>& __x);

  template <class _Tp, class _Up>
  constexpr std::enable_if_t<
      is_convertible_v<decltype(declval<const _Tp&>() != declval<const _Up&>()),
                       bool>,
      bool>
  operator!=(const optional<_Tp>& __x, const _Up& __v);

  template <class _Tp, class _Up>
  constexpr std::enable_if_t<
      is_convertible_v<decltype(declval<const _Tp&>() != declval<const _Up&>()),
                       bool>,
      bool>
  operator!=(const _Tp& __v, const optional<_Up>& __x);

  template <typename T>
  constexpr optional<typename std::decay<T>::type> make_optional(T&& v) {
    return optional<typename std::decay<T>::type>(std::forward<T>(v));
  }

  template <typename T, typename... Args>
  constexpr optional<T> make_optional(Args&&... args) {
    return optional<T>(in_place_t(), std::forward<Args>(args)...);
  }

  template <typename T, typename U, typename... Args>
  constexpr optional<T> make_optional(std::initializer_list<U> il,
                                      Args&&... args) {
    return optional<T>(in_place_t(), il, std::forward<Args>(args)...);
  }

  }  // namespace std

  namespace absl {

#define ABSL_PREDICT_FALSE(x) (__builtin_expect(false || (x), false))
#define ABSL_PREDICT_TRUE(x) (__builtin_expect(false || (x), true))

  struct VLogSite final {
    explicit constexpr VLogSite(const char* f);
    bool IsEnabled(int level);
  };

  namespace log_internal {
  class LogMessage {
   public:
    LogMessage();
    LogMessage& stream();
    LogMessage& InternalStream();
    LogMessage& WithVerbosity(int verboselevel);
    template <typename T>
    LogMessage& operator<<(const T&);
  };
  class LogMessageFatal : public LogMessage {
   public:
    LogMessageFatal();
    ~LogMessageFatal() __attribute__((noreturn));
  };
  class LogMessageQuietlyFatal : public LogMessage {
   public:
    LogMessageQuietlyFatal();
    ~LogMessageQuietlyFatal() __attribute__((noreturn));
  };
  class Voidify final {
   public:
    // This has to be an operator with a precedence lower than << but higher
    // than
    // ?:
    template <typename T>
    void operator&&(const T&) const&& {}
  };
  }  // namespace log_internal
  }  // namespace absl

#ifndef NULL
#define NULL __null
#endif
  extern "C" void abort() {}
#define ABSL_LOG_INTERNAL_LOG_INFO ::absl::log_internal::LogMessage()
#define ABSL_LOG_INTERNAL_LOG_WARNING ::absl::log_internal::LogMessage()
#define ABSL_LOG_INTERNAL_LOG_ERROR ::absl::log_internal::LogMessage()
#define ABSL_LOG_INTERNAL_LOG_FATAL ::absl::log_internal::LogMessageFatal()
#define ABSL_LOG_INTERNAL_LOG_QFATAL \
    ::absl::log_internal::LogMessageQuietlyFatal()
#define LOG(severity) ABSL_LOG_INTERNAL_LOG_##severity.InternalStream()

#define PREDICT_FALSE(x) (__builtin_expect(x, 0))
#define ABSL_LOG_INTERNAL_STRIP_STRING_LITERAL(lit) lit

#define ABSL_LOG_INTERNAL_CHECK(failure_message) ABSL_LOG_INTERNAL_LOG_FATAL
#define ABSL_LOG_INTERNAL_QCHECK(failure_message) ABSL_LOG_INTERNAL_LOG_QFATAL

#define ABSL_LOG_INTERNAL_STATELESS_CONDITION(condition) \
    switch (0)                                             \
    case 0:                                                \
    default:                                               \
      !(condition) ? (void)0 : ::absl::log_internal::Voidify() &&

#define ABSL_LOG_INTERNAL_CONDITION_INFO(type, condition) \
    ABSL_LOG_INTERNAL_##type##_CONDITION(condition)

#define ABSL_LOG_INTERNAL_CONDITION_FATAL(type, condition) \
    ABSL_LOG_INTERNAL_##type##_CONDITION(condition)

#define ABSL_LOG_INTERNAL_CONDITION_QFATAL(type, condition) \
    ABSL_LOG_INTERNAL_##type##_CONDITION(condition)

#define ABSL_CHECK_IMPL(condition, condition_text)                    \
    ABSL_LOG_INTERNAL_CONDITION_FATAL(STATELESS,                        \
                                      ABSL_PREDICT_FALSE(!(condition))) \
    ABSL_LOG_INTERNAL_CHECK(condition_text).InternalStream()

#define ABSL_QCHECK_IMPL(condition, condition_text)                    \
    ABSL_LOG_INTERNAL_CONDITION_QFATAL(STATELESS,                        \
                                       ABSL_PREDICT_FALSE(!(condition))) \
    ABSL_LOG_INTERNAL_QCHECK(condition_text).InternalStream()

#define CHECK(condition) ABSL_CHECK_IMPL((condition), #condition)
#define DCHECK(condition) CHECK(condition)
#define QCHECK(condition) ABSL_QCHECK_IMPL((condition), #condition)

#define ABSL_LOG_INTERNAL_MAX_LOG_VERBOSITY_CHECK(x)

#define VLOG_IS_ON(verbose_level)                               \
    (ABSL_LOG_INTERNAL_MAX_LOG_VERBOSITY_CHECK(verbose_level)[]() \
         ->::absl::VLogSite *                                     \
     {                                                            \
       static ::absl::VLogSite site(__FILE__);                    \
       return &site;                                              \
     }()                                                          \
         ->IsEnabled(verbose_level))

#define ABSL_LOG_IF_IMPL(severity, condition)                 \
    ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, condition) \
        ABSL_LOG_INTERNAL_LOG##severity.InternalStream()

#define ABSL_VLOG_IMPL(verbose_level)                                        \
    switch (const int absl_logging_internal_verbose_level = (verbose_level))   \
    default:                                                                   \
      ABSL_LOG_IF_IMPL(_INFO, VLOG_IS_ON(absl_logging_internal_verbose_level)) \
          .WithVerbosity(absl_logging_internal_verbose_level)

#define VLOG(severity) ABSL_VLOG_IMPL(severity)

  namespace absl {

  template <typename T>
  class StatusOr;
  class Status;

  namespace status_internal {
  std::string* MakeCheckFailString(const absl::Status* status,
                                   const char* prefix);
  }  // namespace status_internal

  namespace log_internal {
  template <class T>
  const T& GetReferenceableValue(const T& t);
  inline char GetReferenceableValue(char t) { return t; }
  inline unsigned char GetReferenceableValue(unsigned char t) { return t; }
  inline signed char GetReferenceableValue(signed char t) { return t; }
  inline short GetReferenceableValue(short t) { return t; }        // NOLINT
  inline unsigned short GetReferenceableValue(unsigned short t) {  // NOLINT
    return t;
  }
  inline int GetReferenceableValue(int t) { return t; }  // NOLINT
  inline unsigned int GetReferenceableValue(unsigned int t) { return t; }
  inline long GetReferenceableValue(long t) { return t; }        // NOLINT
  inline unsigned long GetReferenceableValue(unsigned long t) {  // NOLINT
    return t;
  }
  inline long long GetReferenceableValue(long long t) { return t; }  // NOLINT
  inline unsigned long long GetReferenceableValue(                   // NOLINT
      unsigned long long t) {                                        // NOLINT
    return t;
  }
  inline const absl::Status* AsStatus(const absl::Status& s) { return &s; }
  template <typename T>
  const absl::Status* AsStatus(const absl::StatusOr<T>& s) {
    return &s.status();
  }
  }  // namespace log_internal
  }  // namespace absl
  // TODO(tkd): this still doesn't allow operator<<, unlike the real CHECK_
  // macros.
#define ABSL_LOG_INTERNAL_CHECK_OP(name, op, val1, val2)        \
    while (char* _result = ::absl::log_internal::name##Impl(      \
               ::absl::log_internal::GetReferenceableValue(val1), \
               ::absl::log_internal::GetReferenceableValue(val2), \
               #val1 " " #op " " #val2))                          \
    (void)0
#define ABSL_LOG_INTERNAL_QCHECK_OP(name, op, val1, val2)       \
    while (char* _result = ::absl::log_internal::name##Impl(      \
               ::absl::log_internal::GetReferenceableValue(val1), \
               ::absl::log_internal::GetReferenceableValue(val2), \
               #val1 " " #op " " #val2))                          \
    (void)0
  namespace absl {
  namespace log_internal {
  template <class T1, class T2>
  char* Check_NEImpl(const T1& v1, const T2& v2, const char* names);
  template <class T1, class T2>
  char* Check_EQImpl(const T1& v1, const T2& v2, const char* names);
  template <class T1, class T2>
  char* Check_LTImpl(const T1& v1, const T2& v2, const char* names);

#define CHECK_EQ(a, b) ABSL_LOG_INTERNAL_CHECK_OP(Check_EQ, ==, a, b)
#define CHECK_NE(a, b) ABSL_LOG_INTERNAL_CHECK_OP(Check_NE, !=, a, b)
#define CHECK_LT(a, b) ABSL_LOG_INTERNAL_CHECK_OP(Check_EQ, <, a, b)

#define QCHECK_EQ(a, b) ABSL_LOG_INTERNAL_QCHECK_OP(Check_EQ, ==, a, b)
#define QCHECK_NE(a, b) ABSL_LOG_INTERNAL_QCHECK_OP(Check_NE, !=, a, b)
  }  // namespace log_internal
  }  // namespace absl

#define CHECK_NOTNULL(x) CHECK((x) != nullptr)

#define ABSL_LOG_INTERNAL_CHECK(failure_message) \
    ::absl::log_internal::LogMessageFatal()
#define ABSL_LOG_INTERNAL_QCHECK(failure_message) \
    ::absl::log_internal::LogMessageQuietlyFatal()
#define ABSL_LOG_INTERNAL_CHECK_OK(val)                                      \
    for (::std::pair<const ::absl::Status*, ::std::string*>                    \
             absl_log_internal_check_ok_goo;                                   \
         absl_log_internal_check_ok_goo.first =                                \
             ::absl::log_internal::AsStatus(val),                              \
         absl_log_internal_check_ok_goo.second =                               \
             ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok())     \
                 ? nullptr                                                     \
                 : ::absl::status_internal::MakeCheckFailString(               \
                       absl_log_internal_check_ok_goo.first,                   \
                       ABSL_LOG_INTERNAL_STRIP_STRING_LITERAL(#val " is OK")), \
         !ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok());)      \
    ABSL_LOG_INTERNAL_CHECK(*absl_log_internal_check_ok_goo.second)            \
        .InternalStream()
#define ABSL_LOG_INTERNAL_QCHECK_OK(val)                                     \
    for (::std::pair<const ::absl::Status*, ::std::string*>                    \
             absl_log_internal_check_ok_goo;                                   \
         absl_log_internal_check_ok_goo.first =                                \
             ::absl::log_internal::AsStatus(val),                              \
         absl_log_internal_check_ok_goo.second =                               \
             ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok())     \
                 ? nullptr                                                     \
                 : ::absl::status_internal::MakeCheckFailString(               \
                       absl_log_internal_check_ok_goo.first,                   \
                       ABSL_LOG_INTERNAL_STRIP_STRING_LITERAL(#val " is OK")), \
         !ABSL_PREDICT_TRUE(absl_log_internal_check_ok_goo.first->ok());)      \
    ABSL_LOG_INTERNAL_QCHECK(*absl_log_internal_check_ok_goo.second)           \
        .InternalStream()

#define CHECK_OK(val) ABSL_LOG_INTERNAL_CHECK_OK(val)
#define DCHECK_OK(val) ABSL_LOG_INTERNAL_CHECK_OK(val)
#define QCHECK_OK(val) ABSL_LOG_INTERNAL_QCHECK_OK(val)

  namespace testing {
  struct AssertionResult {
    template <typename T>
    explicit AssertionResult(const T& res, bool enable_if = true) {}
    ~AssertionResult();
    operator bool() const;
    template <typename T>
    AssertionResult& operator<<(const T& value);
    const char* failure_message() const;
  };

  class TestPartResult {
   public:
    enum Type { kSuccess, kNonFatalFailure, kFatalFailure, kSkip };
  };

  class Test {
   public:
    virtual ~Test() = default;

   protected:
    virtual void SetUp() {}
  };

  class Message {
   public:
    template <typename T>
    Message& operator<<(const T& val);
  };

  namespace internal {
  class AssertHelper {
   public:
    AssertHelper(TestPartResult::Type type, const char* file, int line,
                 const char* message);
    void operator=(const Message& message) const;
  };

  class EqHelper {
   public:
    template <typename T1, typename T2>
    static AssertionResult Compare(const char* lhx, const char* rhx,
                                   const T1& lhs, const T2& rhs);
  };

#define GTEST_IMPL_CMP_HELPER_(op_name)                                    \
    template <typename T1, typename T2>                                      \
    AssertionResult CmpHelper##op_name(const char* expr1, const char* expr2, \
                                       const T1& val1, const T2& val2);

  GTEST_IMPL_CMP_HELPER_(NE)
  GTEST_IMPL_CMP_HELPER_(LE)
  GTEST_IMPL_CMP_HELPER_(LT)
  GTEST_IMPL_CMP_HELPER_(GE)
  GTEST_IMPL_CMP_HELPER_(GT)

#undef GTEST_IMPL_CMP_HELPER_

  std::string GetBoolAssertionFailureMessage(
      const AssertionResult& assertion_result, const char* expression_text,
      const char* actual_predicate_value, const char* expected_predicate_value);

  template <typename M>
  class PredicateFormatterFromMatcher {
   public:
    template <typename T>
    AssertionResult operator()(const char* value_text, const T& x) const;
  };

  template <typename M>
  inline PredicateFormatterFromMatcher<M> MakePredicateFormatterFromMatcher(
      M matcher) {
    return PredicateFormatterFromMatcher<M>();
  }
  }  // namespace internal

  namespace status {
  namespace internal_status {
  class IsOkMatcher {};

  class StatusIsMatcher {};

  class CanonicalStatusIsMatcher {};

  template <typename M>
  class IsOkAndHoldsMatcher {};

  }  // namespace internal_status

  internal_status::IsOkMatcher IsOk();

  template <typename StatusCodeMatcher>
  internal_status::StatusIsMatcher StatusIs(StatusCodeMatcher&& code_matcher);

  template <typename StatusCodeMatcher>
  internal_status::CanonicalStatusIsMatcher CanonicalStatusIs(
      StatusCodeMatcher&& code_matcher);

  template <typename InnerMatcher>
  internal_status::IsOkAndHoldsMatcher<InnerMatcher> IsOkAndHolds(
      InnerMatcher m);
  }  // namespace status

  class IsTrueMatcher {};
  IsTrueMatcher IsTrue();

  class IsFalseMatcher {};
  IsFalseMatcher IsFalse();

  }  // namespace testing

  namespace absl_testing {
  namespace status_internal {
  class IsOkMatcher {};
  template <typename M>
  class IsOkAndHoldsMatcher {};
  class StatusIsMatcher {};
  class CanonicalStatusIsMatcher {};
  }  // namespace status_internal
  status_internal::IsOkMatcher IsOk();
  template <typename InnerMatcher>
  status_internal::IsOkAndHoldsMatcher<InnerMatcher> IsOkAndHolds(
      InnerMatcher m);
  template <typename StatusCodeMatcher>
  status_internal::StatusIsMatcher StatusIs(StatusCodeMatcher&& code_matcher);

  template <typename StatusCodeMatcher>
  status_internal::CanonicalStatusIsMatcher CanonicalStatusIs(
      StatusCodeMatcher&& code_matcher);
  }  // namespace absl_testing

  using testing::AssertionResult;
#define EXPECT_TRUE(x)                                          \
    switch (0)                                                    \
    case 0:                                                       \
    default:                                                      \
      if (const AssertionResult gtest_ar_ = AssertionResult(x)) { \
      } else /* NOLINT */                                         \
        ::testing::Message()
#define EXPECT_FALSE(x) EXPECT_TRUE(!(x))

#define GTEST_AMBIGUOUS_ELSE_BLOCKER_ \
    switch (0)                          \
    case 0:                             \
    default:

#define GTEST_ASSERT_(expression, on_failure)                   \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_                                 \
    if (const ::testing::AssertionResult gtest_ar = (expression)) \
      ;                                                           \
    else                                                          \
      on_failure(gtest_ar.failure_message())
#define GTEST_PRED_FORMAT1_(pred_format, v1, on_failure) \
    GTEST_ASSERT_(pred_format(#v1, v1), on_failure)
#define GTEST_PRED_FORMAT2_(pred_format, v1, v2, on_failure) \
    GTEST_ASSERT_(pred_format(#v1, #v2, v1, v2), on_failure)
#define GTEST_MESSAGE_AT_(file, line, message, result_type)             \
    ::testing::internal::AssertHelper(result_type, file, line, message) = \
        ::testing::Message()
#define GTEST_MESSAGE_(message, result_type) \
    GTEST_MESSAGE_AT_(__FILE__, __LINE__, message, result_type)
#define GTEST_FATAL_FAILURE_(message) \
    return GTEST_MESSAGE_(message, ::testing::TestPartResult::kFatalFailure)
#define GTEST_NONFATAL_FAILURE_(message) \
    GTEST_MESSAGE_(message, ::testing::TestPartResult::kNonFatalFailure)

#define ASSERT_PRED_FORMAT1(pred_format, v1) \
    GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_FATAL_FAILURE_)
#define ASSERT_PRED_FORMAT2(pred_format, v1, v2) \
    GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_FATAL_FAILURE_)

#define ASSERT_THAT(value, matcher)                                    \
    ASSERT_PRED_FORMAT1(                                                 \
        ::testing::internal::MakePredicateFormatterFromMatcher(matcher), \
        value)
#define ASSERT_OK(x) ASSERT_THAT(x, ::testing::status::IsOk())

#define EXPECT_PRED_FORMAT1(pred_format, v1) \
    GTEST_PRED_FORMAT1_(pred_format, v1, GTEST_NONFATAL_FAILURE_)
#define EXPECT_PRED_FORMAT2(pred_format, v1, v2) \
    GTEST_PRED_FORMAT2_(pred_format, v1, v2, GTEST_NONFATAL_FAILURE_)
#define EXPECT_THAT(value, matcher)                                    \
    EXPECT_PRED_FORMAT1(                                                 \
        ::testing::internal::MakePredicateFormatterFromMatcher(matcher), \
        value)
#define EXPECT_OK(expression) EXPECT_THAT(expression, ::testing::status::IsOk())

#define GTEST_TEST_BOOLEAN_(expression, text, actual, expected, fail) \
    GTEST_AMBIGUOUS_ELSE_BLOCKER_                                       \
    if (const ::testing::AssertionResult gtest_ar_ =                    \
            ::testing::AssertionResult(expression))                     \
      ;                                                                 \
    else                                                                \
      fail(::testing::internal::GetBoolAssertionFailureMessage(         \
               gtest_ar_, text, #actual, #expected)                     \
               .c_str())
#define GTEST_ASSERT_TRUE(condition) \
    GTEST_TEST_BOOLEAN_(condition, #condition, false, true, GTEST_FATAL_FAILURE_)
#define GTEST_ASSERT_FALSE(condition)                        \
    GTEST_TEST_BOOLEAN_(!(condition), #condition, true, false, \
                        GTEST_FATAL_FAILURE_)
#define ASSERT_TRUE(condition) GTEST_ASSERT_TRUE(condition)
#define ASSERT_FALSE(condition) GTEST_ASSERT_FALSE(condition)

#define EXPECT_EQ(x, y) \
    EXPECT_PRED_FORMAT2(::testing::internal::EqHelper::Compare, x, y)
#define EXPECT_NE(x, y) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperNE, x, y)
#define EXPECT_LT(x, y) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLT, x, y)
#define EXPECT_GT(x, y) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGT, x, y)
#define EXPECT_LE(x, y) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperLE, x, y)
#define EXPECT_GE(x, y) \
    EXPECT_PRED_FORMAT2(::testing::internal::CmpHelperGE, x, y)

#define ASSERT_EQ(x, y) \
    ASSERT_PRED_FORMAT2(testing::internal::EqHelper::Compare, x, y)
#define ASSERT_NE(x, y) \
    ASSERT_PRED_FORMAT2(testing::internal::CmpHelperNE, x, y)
#define ASSERT_LT(x, y) \
    ASSERT_PRED_FORMAT2(testing::internal::CmpHelperLT, x, y)
#define ASSERT_GT(x, y) \
    ASSERT_PRED_FORMAT2(testing::internal::CmpHelperGT, x, y)
#define ASSERT_LE(x, y) \
    ASSERT_PRED_FORMAT2(testing::internal::CmpHelperLE, x, y)
#define ASSERT_GE(x, y) \
    ASSERT_PRED_FORMAT2(testing::internal::CmpHelperGE, x, y)

#endif  // BASE_DEFS_H_
)cc";

constexpr const char *kAbslTypeTraitsHeader = R"cc(
#pragma clang system_header

#ifndef ABSL_TYPE_TRAITS_H_
#define ABSL_TYPE_TRAITS_H_

#include "std_type_traits.h"

  namespace absl {

  template <typename... Ts>
  struct conjunction : std::true_type {};

  template <typename T, typename... Ts>
  struct conjunction<T, Ts...>
      : std::conditional<T::value, conjunction<Ts...>, T>::type {};

  template <typename T>
  struct conjunction<T> : T {};

  template <typename... Ts>
  struct disjunction : std::false_type {};

  template <typename T, typename... Ts>
  struct disjunction<T, Ts...>
      : std::conditional<T::value, T, disjunction<Ts...>>::type {};

  template <typename T>
  struct disjunction<T> : T {};

  template <typename T>
  struct negation : std::integral_constant<bool, !T::value> {};

  template <typename T>
  using remove_cv_t = typename std::remove_cv<T>::type;

  template <typename T>
  using remove_reference_t = typename std::remove_reference<T>::type;

  template <typename T>
  using decay_t = typename std::decay<T>::type;

  template <bool B, typename T = void>
  using enable_if_t = typename std::enable_if<B, T>::type;

  template <bool B, typename T, typename F>
  using conditional_t = typename std::conditional<B, T, F>::type;

  }  // namespace absl

#endif  // ABSL_TYPE_TRAITS_H_
)cc";

constexpr const char *kStdTypeTraitsHeader = R"cc(
#pragma clang system_header

#ifndef STD_TYPE_TRAITS_H_
#define STD_TYPE_TRAITS_H_

#include "stdlib_defs.h"

  namespace std {

  template <bool B, class T, class F>
  struct conditional {
    typedef T type;
  };

  template <class T, class F>
  struct conditional<false, T, F> {
    typedef F type;
  };

  template <bool B, class T, class F>
  using conditional_t = typename conditional<B, T, F>::type;

  template <class...>
  struct conjunction : true_type {};

  template <class B1>
  struct conjunction<B1> : B1 {};

  template <class B1, class... Bn>
  struct conjunction<B1, Bn...>
      : conditional_t<bool(B1::value), conjunction<Bn...>, B1> {};

  template <bool B, class T = void>
  struct enable_if {};

  template <class T>
  struct enable_if<true, T> {
    typedef T type;
  };

  template <bool B, class T = void>
  using enable_if_t = typename enable_if<B, T>::type;

  template <class T, class U>
  struct is_same : false_type {};

  template <class T>
  struct is_same<T, T> : true_type {};

  template <class T, class U>
  inline constexpr bool is_same_v = is_same<T, U>::value;

  template <class T>
  struct is_array : false_type {};

  template <class T>
  struct is_array<T[]> : true_type {};

  template <class T, size_t N>
  struct is_array<T[N]> : true_type {};

  template <class T>
  struct remove_extent {
    typedef T type;
  };

  template <class T>
  struct remove_extent<T[]> {
    typedef T type;
  };

  template <class T, size_t N>
  struct remove_extent<T[N]> {
    typedef T type;
  };

  // primary template
  template <class>
  struct is_function : false_type {};

  // specialization for regular functions
  template <class Ret, class... Args>
  struct is_function<Ret(Args...)> : true_type {};

  namespace detail {

  template <class T>
  struct type_identity {
    using type = T;
  };  // or use type_identity (since C++20)

  template <class T>
  auto try_add_pointer(int)
      -> type_identity<typename remove_reference<T>::type*>;
  template <class T>
  auto try_add_pointer(...) -> type_identity<T>;

  }  // namespace detail

  template <class T>
  struct add_pointer : decltype(detail::try_add_pointer<T>(0)) {};

  template <class T>
  struct remove_cv {
    typedef T type;
  };
  template <class T>
  struct remove_cv<const T> {
    typedef T type;
  };
  template <class T>
  struct remove_cv<volatile T> {
    typedef T type;
  };
  template <class T>
  struct remove_cv<const volatile T> {
    typedef T type;
  };

  template <class T>
  struct remove_const {
    typedef T type;
  };
  template <class T>
  struct remove_const<const T> {
    typedef T type;
  };

  template <class T>
  struct remove_volatile {
    typedef T type;
  };
  template <class T>
  struct remove_volatile<volatile T> {
    typedef T type;
  };

  template <class T>
  using remove_cv_t = typename remove_cv<T>::type;

  template <class T>
  using remove_const_t = typename remove_const<T>::type;

  template <class T>
  using remove_volatile_t = typename remove_volatile<T>::type;

  template <class T>
  struct decay {
   private:
    typedef typename remove_reference<T>::type U;

   public:
    typedef typename conditional<
        is_array<U>::value, typename remove_extent<U>::type*,
        typename conditional<is_function<U>::value,
                             typename add_pointer<U>::type,
                             typename remove_cv<U>::type>::type>::type type;
  };

  namespace detail {

  template <class T>  // Note that `cv void&` is a substitution failure
  auto try_add_lvalue_reference(int) -> type_identity<T&>;
  template <class T>  // Handle T = cv void case
  auto try_add_lvalue_reference(...) -> type_identity<T>;

  template <class T>
  auto try_add_rvalue_reference(int) -> type_identity<T&&>;
  template <class T>
  auto try_add_rvalue_reference(...) -> type_identity<T>;

  }  // namespace detail

  template <class T>
  struct add_lvalue_reference
      : decltype(detail::try_add_lvalue_reference<T>(0)) {};

  template <class T>
  struct add_rvalue_reference
      : decltype(detail::try_add_rvalue_reference<T>(0)) {};

  template <class T>
  typename add_rvalue_reference<T>::type declval() noexcept;

  template <class T>
  struct is_void : is_same<void, typename remove_cv<T>::type> {};

  namespace detail {

  template <class T>
  auto test_returnable(int)
      -> decltype(void(static_cast<T (*)()>(nullptr)), true_type{});
  template <class>
  auto test_returnable(...) -> false_type;

  template <class From, class To>
  auto test_implicitly_convertible(int)
      -> decltype(void(declval<void (&)(To)>()(declval<From>())), true_type{});
  template <class, class>
  auto test_implicitly_convertible(...) -> false_type;

  }  // namespace detail

  template <class From, class To>
  struct is_convertible
      : integral_constant<
            bool, (decltype(detail::test_returnable<To>(0))::value &&
                   decltype(detail::test_implicitly_convertible<From, To>(
                       0))::value) ||
                      (is_void<From>::value && is_void<To>::value)> {};

  template <class From, class To>
  inline constexpr bool is_convertible_v = is_convertible<From, To>::value;

  template <class...>
  using void_t = void;

  template <class, class T, class... Args>
  struct is_constructible_ : false_type {};

  template <class T, class... Args>
  struct is_constructible_<void_t<decltype(T(declval<Args>()...))>, T, Args...>
      : true_type {};

  template <class T, class... Args>
  using is_constructible = is_constructible_<void_t<>, T, Args...>;

  template <class T, class... Args>
  inline constexpr bool is_constructible_v =
      is_constructible<T, Args...>::value;

  template <class T>
  struct is_reference : false_type {};
  template <class T>
  struct is_reference<T&> : true_type {};
  template <class T>
  struct is_reference<T&&> : true_type {};

  template <class T>
  inline constexpr bool is_reference_v = is_reference<T>::value;

  template <class _Tp>
  struct __uncvref {
    typedef typename remove_cv<typename remove_reference<_Tp>::type>::type type;
  };

  template <class _Tp>
  using __uncvref_t = typename __uncvref<_Tp>::type;

  template <bool _Val>
  using _BoolConstant = integral_constant<bool, _Val>;

  template <class _Tp, class _Up>
  using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;

  template <class _Tp, class _Up>
  using _IsNotSame = _BoolConstant<!__is_same(_Tp, _Up)>;

  template <bool>
  struct _MetaBase;
  template <>
  struct _MetaBase<true> {
    template <class _Tp, class _Up>
    using _SelectImpl = _Tp;
    template <template <class...> class _FirstFn, template <class...> class,
              class... _Args>
    using _SelectApplyImpl = _FirstFn<_Args...>;
    template <class _First, class...>
    using _FirstImpl = _First;
    template <class, class _Second, class...>
    using _SecondImpl = _Second;
    template <class _Result, class _First, class... _Rest>
    using _OrImpl =
        typename _MetaBase<_First::value != true && sizeof...(_Rest) != 0>::
            template _OrImpl<_First, _Rest...>;
  };

  template <>
  struct _MetaBase<false> {
    template <class _Tp, class _Up>
    using _SelectImpl = _Up;
    template <template <class...> class, template <class...> class _SecondFn,
              class... _Args>
    using _SelectApplyImpl = _SecondFn<_Args...>;
    template <class _Result, class...>
    using _OrImpl = _Result;
  };

  template <bool _Cond, class _IfRes, class _ElseRes>
  using _If = typename _MetaBase<_Cond>::template _SelectImpl<_IfRes, _ElseRes>;

  template <class... _Rest>
  using _Or = typename _MetaBase<sizeof...(_Rest) !=
                                 0>::template _OrImpl<false_type, _Rest...>;

  template <bool _Bp, class _Tp = void>
  using __enable_if_t = typename enable_if<_Bp, _Tp>::type;

  template <class...>
  using __expand_to_true = true_type;
  template <class... _Pred>
  __expand_to_true<__enable_if_t<_Pred::value>...> __and_helper(int);
  template <class...>
  false_type __and_helper(...);
  template <class... _Pred>
  using _And = decltype(__and_helper<_Pred...>(0));

  template <class _Pred>
  struct _Not : _BoolConstant<!_Pred::value> {};

  struct __check_tuple_constructor_fail {
    static constexpr bool __enable_explicit_default() { return false; }
    static constexpr bool __enable_implicit_default() { return false; }
    template <class...>
    static constexpr bool __enable_explicit() {
      return false;
    }
    template <class...>
    static constexpr bool __enable_implicit() {
      return false;
    }
  };

  template <typename, typename _Tp>
  struct __select_2nd {
    typedef _Tp type;
  };
  template <class _Tp, class _Arg>
  typename __select_2nd<decltype((declval<_Tp>() = declval<_Arg>())),
                        true_type>::type
  __is_assignable_test(int);
  template <class, class>
  false_type __is_assignable_test(...);
  template <class _Tp, class _Arg,
            bool = is_void<_Tp>::value || is_void<_Arg>::value>
  struct __is_assignable_imp
      : public decltype((__is_assignable_test<_Tp, _Arg>(0))) {};
  template <class _Tp, class _Arg>
  struct __is_assignable_imp<_Tp, _Arg, true> : public false_type {};
  template <class _Tp, class _Arg>
  struct is_assignable : public __is_assignable_imp<_Tp, _Arg> {};

  template <class _Tp>
  struct __libcpp_is_integral : public false_type {};
  template <>
  struct __libcpp_is_integral<bool> : public true_type {};
  template <>
  struct __libcpp_is_integral<char> : public true_type {};
  template <>
  struct __libcpp_is_integral<signed char> : public true_type {};
  template <>
  struct __libcpp_is_integral<unsigned char> : public true_type {};
  template <>
  struct __libcpp_is_integral<wchar_t> : public true_type {};
  template <>
  struct __libcpp_is_integral<short> : public true_type {};  // NOLINT
  template <>
  struct __libcpp_is_integral<unsigned short> : public true_type {};  // NOLINT
  template <>
  struct __libcpp_is_integral<int> : public true_type {};
  template <>
  struct __libcpp_is_integral<unsigned int> : public true_type {};
  template <>
  struct __libcpp_is_integral<long> : public true_type {};  // NOLINT
  template <>
  struct __libcpp_is_integral<unsigned long> : public true_type {};  // NOLINT
  template <>
  struct __libcpp_is_integral<long long> : public true_type {};  // NOLINT
  template <>  // NOLINTNEXTLINE
  struct __libcpp_is_integral<unsigned long long> : public true_type {};
  template <class _Tp>
  struct is_integral
      : public __libcpp_is_integral<typename remove_cv<_Tp>::type> {};

  template <class _Tp>
  struct __libcpp_is_floating_point : public false_type {};
  template <>
  struct __libcpp_is_floating_point<float> : public true_type {};
  template <>
  struct __libcpp_is_floating_point<double> : public true_type {};
  template <>
  struct __libcpp_is_floating_point<long double> : public true_type {};
  template <class _Tp>
  struct is_floating_point
      : public __libcpp_is_floating_point<typename remove_cv<_Tp>::type> {};

  template <class _Tp>
  struct is_arithmetic
      : public integral_constant<bool, is_integral<_Tp>::value ||
                                           is_floating_point<_Tp>::value> {};

  template <class _Tp>
  struct __libcpp_is_pointer : public false_type {};
  template <class _Tp>
  struct __libcpp_is_pointer<_Tp*> : public true_type {};
  template <class _Tp>
  struct is_pointer
      : public __libcpp_is_pointer<typename remove_cv<_Tp>::type> {};

  template <class _Tp>
  struct __libcpp_is_member_pointer : public false_type {};
  template <class _Tp, class _Up>
  struct __libcpp_is_member_pointer<_Tp _Up::*> : public true_type {};
  template <class _Tp>
  struct is_member_pointer
      : public __libcpp_is_member_pointer<typename remove_cv<_Tp>::type> {};

  template <class _Tp>
  struct __libcpp_union : public false_type {};
  template <class _Tp>
  struct is_union : public __libcpp_union<typename remove_cv<_Tp>::type> {};

  struct __two {
    char __lx[2];
  };

  namespace __is_class_imp {
  template <class _Tp>
  char __test(int _Tp::*);
  template <class _Tp>
  __two __test(...);
  }  // namespace __is_class_imp
  template <class _Tp>
  struct is_class
      : public integral_constant<bool,
                                 sizeof(__is_class_imp::__test<_Tp>(0)) == 1 &&
                                     !is_union<_Tp>::value> {};

  template <class _Tp>
  struct __is_nullptr_t_impl : public false_type {};
  template <>
  struct __is_nullptr_t_impl<nullptr_t> : public true_type {};
  template <class _Tp>
  struct __is_nullptr_t
      : public __is_nullptr_t_impl<typename remove_cv<_Tp>::type> {};
  template <class _Tp>
  struct is_null_pointer
      : public __is_nullptr_t_impl<typename remove_cv<_Tp>::type> {};

  template <class _Tp>
  struct is_enum
      : public integral_constant<
            bool, !is_void<_Tp>::value && !is_integral<_Tp>::value &&
                      !is_floating_point<_Tp>::value && !is_array<_Tp>::value &&
                      !is_pointer<_Tp>::value && !is_reference<_Tp>::value &&
                      !is_member_pointer<_Tp>::value && !is_union<_Tp>::value &&
                      !is_class<_Tp>::value && !is_function<_Tp>::value> {};

  template <class _Tp>
  struct is_scalar
      : public integral_constant<
            bool, is_arithmetic<_Tp>::value || is_member_pointer<_Tp>::value ||
                      is_pointer<_Tp>::value || __is_nullptr_t<_Tp>::value ||
                      is_enum<_Tp>::value> {};
  template <>
  struct is_scalar<nullptr_t> : public true_type {};

  }  // namespace std

#endif  // STD_TYPE_TRAITS_H_
)cc";

constexpr const char *kStdLibDefsHeader = R"cc(
#pragma clang system_header

#ifndef STDLIB_DEFS_H_
#define STDLIB_DEFS_H_

  // mock definitions of entities in the C++ standard library.
  //
  // mocks are only defined to the extent needed. Feel free to extend their
  // definitions to suit your new tests.
  //
  // If you find that you need to break up this file into smaller files then
  // break along the boundaries of the stdlib's header files.

  // From stddef.h
  typedef decltype(sizeof(char)) size_t;      // NOLINT
  typedef decltype(sizeof(char*)) ptrdiff_t;  // NOLINT

  // From stdint.h
  typedef __SIZE_TYPE__ size_t;
  typedef __UINT8_TYPE__ uint8_t;
  typedef __UINT16_TYPE__ uint16_t;
  typedef __UINT32_TYPE__ uint32_t;
  typedef __UINT64_TYPE__ uint64_t;
  typedef __INT8_TYPE__ int8_t;
  typedef __INT16_TYPE__ int16_t;
  typedef __INT32_TYPE__ int32_t;
  typedef __INT64_TYPE__ int64_t;

  namespace std {
  // Our implementation of the standard library puts everything inside namespace
  // std into an inline namespace, whose name should not be hardcoded in
  // ClangTidy.
  inline namespace __do_not_hardcode_in_clang_tidy {

  using size_t = ::size_t;

  template <typename T, T V>
  struct integral_constant {
    static constexpr T value = V;
  };
  using true_type = integral_constant<bool, true>;
  using false_type = integral_constant<bool, false>;

  using nullptr_t = decltype(nullptr);

  template <typename T>
  struct _NonDeducedImpl {
    using type = T;
  };
  template <typename T>
  using _NonDeduced = typename _NonDeducedImpl<T>::type;

  // clang-format off
template <typename T> struct remove_reference      { using type = T; };
template <typename T> struct remove_reference<T&>  { using type = T; };
template <typename T> struct remove_reference<T&&> { using type = T; };
  // clang-format on

  template <typename T>
  using remove_reference_t = typename remove_reference<T>::type;

  template <typename T>
  constexpr T&& forward(remove_reference_t<T>& t) noexcept;

  template <typename T>
  constexpr T&& forward(remove_reference_t<T>&& t) noexcept;

  template <typename T>
  void swap(T& a1, T& a2);

  template <typename F>
  class function {
   public:
    function();
    function(const function&);
    function(function&&);
    function& operator=(const function&);
    function& operator=(function&&);
    ~function();
  };

  template <typename T>
  class initializer_list {
   public:
    using size_type = decltype(sizeof(0));
    const T* begin() const;
    const T* end() const;

   private:
    constexpr initializer_list(const T* items, size_type size)
        : items_(items), size_(size) {}
    const T* items_;
    size_type size_;
  };

  // <functional>
  template <class T>
  struct equal_to {
    bool operator()(const T& lhs, const T& rhs) const;
  };

  template <class T>
  struct not_equal_to {
    bool operator()(const T& lhs, const T& rhs) const;
  };

  template <class T>
  struct hash {
    std::size_t operator()(T const& t) const noexcept;
  };

  template <class T = void>
  struct less {
    constexpr bool operator()(const T& lhs, const T& rhs) const;
  };

  template <>
  struct less<void> {
    using is_transparent = void;
    template <typename T, typename U>
    constexpr auto operator()(T&& lhs, U&& rhs) const
        -> decltype(std::forward<T>(lhs) < std::forward<U>(rhs));
  };

  template <class T>
  struct greater {
    bool operator()(const T& lhs, const T& rhs) const;
  };

  // <algorithm>
  template <class T>
  const T& min(const T& a, const T& b);
  template <class T, class Compare>
  const T& min(const T& a, const T& b, Compare comp);
  template <class T>
  const T& max(const T& a, const T& b);
  template <class T, class Compare>
  const T& max(const T& a, const T& b, Compare comp);
  template <class T>
  T min(std::initializer_list<T> ilist);
  template <class T, class Compare>
  T min(std::initializer_list<T> ilist, Compare comp);
  template <class T>
  T max(std::initializer_list<T> ilist);
  template <class T, class Compare>
  T max(std::initializer_list<T> ilist, Compare comp);

  // <memory>
  template <class T>
  struct allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef T value_type;

    T* allocate(size_t n);
  };

  template <class Alloc>
  struct allocator_traits {
    typedef Alloc allocator_type;
    typedef typename allocator_type::value_type value_type;
    typedef typename allocator_type::pointer pointer;
    typedef typename allocator_type::const_pointer const_pointer;
    typedef typename allocator_type::difference_type difference_type;
    typedef typename allocator_type::size_type size_type;
  };

  template <class... T>
  class tuple {};

  template <std::size_t I, class T>
  struct tuple_element;

  template <std::size_t I, class T, class... Ts>
  struct tuple_element<I, std::tuple<T, Ts...>>
      : std::tuple_element<I - 1, std::tuple<Ts...>> {};

  template <class T, class... Ts>
  struct tuple_element<0, std::tuple<T, Ts...>> {
    using type = T;
  };

  template <std::size_t I, typename T>
  using tuple_element_t = typename tuple_element<I, T>::type;

  template <class T1, class T2>
  struct pair {
    T1 first;
    T2 second;

    typedef T1 first_type;
    typedef T2 second_type;

    constexpr pair();

    template <class U1, class U2>
    pair(pair<U1, U2>&& p);

    template <class U1, class U2>
    pair(U1&& x, U2&& y);
  };

  template <class T1, class T2>
  pair<T1, T2> make_pair(T1&& t1, T2&& t2);

  template <class InputIt1, class InputIt2, class OutputIt>
  OutputIt set_difference(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                          InputIt2 last2, OutputIt d_first);

  template <class InputIt1, class InputIt2, class OutputIt>
  OutputIt set_intersection(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                            InputIt2 last2, OutputIt d_first);

  template <class InputIt1, class InputIt2, class OutputIt>
  OutputIt set_union(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                     InputIt2 last2, OutputIt d_first);

  template <class InputIt1, class InputIt2, class OutputIt>
  OutputIt set_symmetric_difference(InputIt1 first1, InputIt1 last1,
                                    InputIt2 first2, InputIt2 last2,
                                    OutputIt d_first);

  template <class InputIt1, class InputIt2, class OutputIt>
  OutputIt merge(InputIt1 first1, InputIt1 last1, InputIt2 first2,
                 InputIt2 last2, OutputIt d_first);

  template <typename T, class Allocator = allocator<T>>
  class vector {
   public:
    using value_type = T;
    using size_type = typename allocator_traits<Allocator>::size_type;

    // Constructors.
    vector() {}
    vector(size_type, const Allocator& = Allocator()) {}
    vector(initializer_list<T> initializer_list,
           const Allocator& = Allocator()) {}
    vector(const vector& vector) {}
    ~vector();

    // Modifiers.
    void push_back(const T& value);
    void push_back(T&& value);
    template <typename... Args>
    T& emplace_back(Args&&... args);

    // Iterators
    class InputIterator {
     public:
      InputIterator(const InputIterator&);
      ~InputIterator();
      InputIterator& operator=(const InputIterator&);
      InputIterator& operator++();
      T& operator*() const;
      bool operator!=(const InputIterator&) const;
      bool operator==(const InputIterator&) const;
    };
    typedef InputIterator iterator;
    typedef const InputIterator const_iterator;
    iterator begin() noexcept;
    const_iterator begin() const noexcept;
    const_iterator cbegin() const noexcept;
    iterator end() noexcept;
    const_iterator end() const noexcept;
    const_iterator cend() const noexcept;
    T* data() noexcept;
    const T* data() const noexcept;
    T& operator[](int n);
    const T& operator[](int n) const;
    T& at(int n);
    const T& at(int n) const;
    size_t size() const;
  };

  template <typename T>
  struct default_delete {};

  template <typename T, typename D = default_delete<T>>
  class unique_ptr {
   public:
    using element_type = T;
    using deleter_type = D;

    constexpr unique_ptr();
    constexpr unique_ptr(nullptr_t) noexcept;
    unique_ptr(unique_ptr&&);
    explicit unique_ptr(T*);
    template <typename U, typename E>
    unique_ptr(unique_ptr<U, E>&&);

    ~unique_ptr();

    unique_ptr& operator=(unique_ptr&&);
    template <typename U, typename E>
    unique_ptr& operator=(unique_ptr<U, E>&&);
    unique_ptr& operator=(nullptr_t);

    void reset(T* = nullptr) noexcept;
    T* release();
    T* get() const;

    T& operator*() const;
    T* operator->() const;
    explicit operator bool() const noexcept;
  };

  template <typename T, typename D>
  class unique_ptr<T[], D> {
   public:
    T* get() const;
    T& operator[](size_t i);
    const T& operator[](size_t i) const;
  };

  template <typename T, typename... Args>
  unique_ptr<T> make_unique(Args&&...);

  template <class T, class D>
  void swap(unique_ptr<T, D>& x, unique_ptr<T, D>& y) noexcept;

  template <class T1, class D1, class T2, class D2>
  bool operator==(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);
  template <class T1, class D1, class T2, class D2>
  bool operator!=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);
  template <class T1, class D1, class T2, class D2>
  bool operator<(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);
  template <class T1, class D1, class T2, class D2>
  bool operator<=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);
  template <class T1, class D1, class T2, class D2>
  bool operator>(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);
  template <class T1, class D1, class T2, class D2>
  bool operator>=(const unique_ptr<T1, D1>& x, const unique_ptr<T2, D2>& y);

  template <class T, class D>
  bool operator==(const unique_ptr<T, D>& x, nullptr_t) noexcept;
  template <class T, class D>
  bool operator==(nullptr_t, const unique_ptr<T, D>& y) noexcept;
  template <class T, class D>
  bool operator!=(const unique_ptr<T, D>& x, nullptr_t) noexcept;
  template <class T, class D>
  bool operator!=(nullptr_t, const unique_ptr<T, D>& y) noexcept;
  template <class T, class D>
  bool operator<(const unique_ptr<T, D>& x, nullptr_t);
  template <class T, class D>
  bool operator<(nullptr_t, const unique_ptr<T, D>& y);
  template <class T, class D>
  bool operator<=(const unique_ptr<T, D>& x, nullptr_t);
  template <class T, class D>
  bool operator<=(nullptr_t, const unique_ptr<T, D>& y);
  template <class T, class D>
  bool operator>(const unique_ptr<T, D>& x, nullptr_t);
  template <class T, class D>
  bool operator>(nullptr_t, const unique_ptr<T, D>& y);
  template <class T, class D>
  bool operator>=(const unique_ptr<T, D>& x, nullptr_t);
  template <class T, class D>
  bool operator>=(nullptr_t, const unique_ptr<T, D>& y);

  template <typename T>
  class shared_ptr_base {
   public:
    ~shared_ptr_base();
    T& operator*() const;
    T* operator->() const;
  };

  template <typename T>
  class shared_ptr : public shared_ptr_base<T> {
   public:
    constexpr shared_ptr();
    ~shared_ptr();
  };

  template <class T, class... Args>
  shared_ptr<T> make_shared(Args&&... args);

  template <typename T>
  class weak_ptr {
   public:
    constexpr weak_ptr() noexcept;
    ~weak_ptr();
  };

  template <typename T>
  class reference_wrapper {
   public:
    reference_wrapper(T& ref);
    T& get() const noexcept;
    operator T&() const noexcept;
  };

  template <typename Container>
  typename Container::iterator begin(Container&);
  template <typename Container>
  typename Container::iterator end(Container&);

  template <class T>
  reference_wrapper<const T> cref(const T& t);

  template <typename T>
  constexpr std::remove_reference_t<T>&& move(T&& x);

  struct nothrow_t {
    explicit nothrow_t() = default;
  };
  extern const std::nothrow_t nothrow;

  }  // namespace __do_not_hardcode_in_clang_tidy
  }  // namespace std

#endif  // STDLIB_DEFS_H_
)cc";

constexpr const char *kStringDefsHeader = R"cc(
  // string_defs.h
  //
  // This file contains minimal mock declarations of the string type and
  // friends. It can be used in tests to use matchers as they would be used in a
  // prod google3 environment. This only contains what has been necessary for
  // the tests that use it. It can be expanded if tests need more of it.

#pragma clang system_header

#ifndef STRING_DEFS_H_
#define STRING_DEFS_H_

#include "std_char_traits.h"
#include "std_string_view.h"
#include "std_type_traits.h"
#include "stdlib_defs.h"

#define ABSL_PRINTF_ATTRIBUTE(a, b) __attribute__((format(printf, a, b)))

  namespace __gnu_cxx {
  inline namespace __do_not_hardcode_in_clang_tidy {
  template <typename A, typename B, typename C>
  class __string_base {};
  }  // namespace __do_not_hardcode_in_clang_tidy
  }  // namespace __gnu_cxx

  namespace std {
  inline namespace __do_not_hardcode_in_clang_tidy {

  template <class _CharT, class _Traits, class _Tp>
  struct __can_be_converted_to_string_view {
    static std::true_type Test(std::basic_string_view<_CharT, _Traits>);
    static std::false_type Test(...);
    static const _Tp& Instance();
    static constexpr bool value = decltype(Test(Instance()))::value;
  };

  template <typename A, typename B = std::char_traits<A>,
            typename C = std::allocator<A>>
  class basic_string : public __gnu_cxx::__string_base<A, B, C> {
   public:
    basic_string();
    basic_string(const basic_string&);
    basic_string(basic_string&&) noexcept;
    basic_string(const A*, const C& = C());
    basic_string(const A*, int, const C& = C());
    basic_string(const basic_string&, int, int, C = C());

    template <class _Tp, class = enable_if_t<__can_be_converted_to_string_view<
                             A, B, _Tp>::value>>
    explicit basic_string(const _Tp& __t);

    template <class _Tp, class = enable_if_t<__can_be_converted_to_string_view<
                             A, B, _Tp>::value>>
    explicit basic_string(const _Tp& __t, const C& __a);

    ~basic_string();

    basic_string& operator=(const basic_string&);
    basic_string& operator=(basic_string&&) noexcept;
    basic_string& operator=(const A*);
    template <class _Tp, class = enable_if_t<__can_be_converted_to_string_view<
                             A, B, _Tp>::value>>
    basic_string& operator=(const _Tp& __t);

    basic_string& operator+=(const basic_string&);

    static const int npos = -1;
    const char* c_str() const;
    const char* data() const;
    char* data();
    size_t size() const;
    size_t capacity() const;
    size_t length() const;
    bool empty() const;
    char& operator[](int);
    const char& operator[](int) const;
    char& at(int);
    const char& at(int) const;
    void clear();
    void resize(int);
    int compare(const basic_string&) const;
    int find(const basic_string&, int pos = 0) const;
    int find(const char*, int pos = 0) const;
    int rfind(const basic_string&, int pos = 0) const;
    int rfind(const char*, int pos = 0) const;

    operator std::basic_string_view<A, B>() const noexcept;
  };

  template <typename A, typename B, typename C>
  basic_string<A, B, C> operator+(const basic_string<A, B, C>&,
                                  const basic_string<A, B, C>&);
  template <typename A, typename B, typename C>
  basic_string<A, B, C> operator+(const basic_string<A, B, C>&, const char*);

  typedef basic_string<char> string;

  bool operator==(const string&, const string&);
  bool operator==(const string&, const char*);
  bool operator==(const char*, const string&);

  bool operator!=(const string&, const string&);
  bool operator<(const string&, const string&);
  bool operator>(const string&, const string&);
  bool operator<=(const string&, const string&);
  bool operator>=(const string&, const string&);
  }  // namespace __do_not_hardcode_in_clang_tidy
  }  // namespace std

#endif  // STRING_DEFS_H_
)cc";

constexpr const char *kStdCharTraitsHeader = R"cc(
#ifndef STD_CHAR_TRAITS_H_
#define STD_CHAR_TRAITS_H_

  namespace std {
  inline namespace __do_not_hardcode_in_clang_tidy {
  template <typename T>
  class char_traits {};
  }  // namespace __do_not_hardcode_in_clang_tidy
  }  // namespace std

#endif  // STD_CHAR_TRAITS_H_
)cc";

constexpr const char *kStdStringViewHeader = R"cc(
  // This file is a mock of the <string_view> standard library header.

#pragma clang system_header

#ifndef STD_STRING_VIEW_H_
#define STD_STRING_VIEW_H_

#include "std_char_traits.h"
#include "stdlib_defs.h"

  constexpr size_t strlen(const char* p) {
    return *p ? strlen(p + 1) + 1 : 0;
  }

  namespace std {
  inline namespace __do_not_hardcode_in_clang_tidy {

  template <typename CharT, typename Traits = std::char_traits<CharT>>
  class basic_string_view {
   public:
    using traits_type = Traits;

    constexpr basic_string_view() noexcept;
    constexpr basic_string_view(const CharT* null_terminated_string) {  // NOLINT
      (void)strlen(null_terminated_string);  // some checks need to see strlen.
    }
    constexpr basic_string_view(const CharT*, size_t);

#if __cplusplus >= 202002L
    // A subset of:
    // template <contiguous_iterator _It, sized_sentinel_for<_It> _End>
    //   requires(is_same_v<iter_value_t<_It>, _CharT> &&
    //            !is_convertible_v<_End, size_type>)
    // constexpr basic_string_view(_It __begin, _End __end);
    // to avoid having to define all the underlying concepts, etc.
    constexpr basic_string_view(const CharT* __begin, const CharT* __end);
#endif

    constexpr const char* data() const noexcept;
    constexpr size_t size() const noexcept;
    constexpr size_t length() const noexcept;
    constexpr const CharT& operator[](size_t pos) const;
    constexpr const CharT& at(size_t pos) const;
  };

  using string_view = basic_string_view<char>;

  template <class CharT, class Traits>
  constexpr bool operator==(basic_string_view<CharT, Traits> lhs,
                            basic_string_view<CharT, Traits> rhs) noexcept;
  template <class CharT, class Traits>
  constexpr bool operator==(_NonDeduced<basic_string_view<CharT, Traits>> lhs,
                            basic_string_view<CharT, Traits> rhs) noexcept;
  template <class CharT, class Traits>
  constexpr bool operator==(
      basic_string_view<CharT, Traits> lhs,
      _NonDeduced<basic_string_view<CharT, Traits>> rhs) noexcept;

  }  // namespace __do_not_hardcode_in_clang_tidy
  }  // namespace std

#endif  // STD_STRING_VIEW_H_
)cc";

constexpr const char *kStatusOrDefsHeader =
    R"cc(
#ifndef STATUSOR_H_
#define STATUSOR_H_

#include "absl_defs.h"
#include "std_string_view.h"
#include "stdlib_defs.h"
#include "string_defs.h"

  namespace absl {
  template <typename>
  class AnyInvocable;
  template <typename R, typename... A>
  class AnyInvocable<R(A...) &&> {
   public:
    AnyInvocable();
    ~AnyInvocable();
    template <typename F>
    AnyInvocable(F&& f, decltype(f(A{}...), 0) = 0);

    R operator()(A...) &&;
  };
  }  // namespace absl

  namespace absl {
  struct SourceLocation {
    static constexpr SourceLocation current();
    static constexpr SourceLocation DoNotInvokeDirectlyNoSeriouslyDont(
        int line, const char* file_name);
  };
  }  // namespace absl
  namespace absl {
  enum class StatusCode : int {
    kOk,
    kCancelled,
    kUnknown,
    kInvalidArgument,
    kDeadlineExceeded,
    kNotFound,
    kAlreadyExists,
    kPermissionDenied,
    kResourceExhausted,
    kFailedPrecondition,
    kAborted,
    kOutOfRange,
    kUnimplemented,
    kInternal,
    kUnavailable,
    kDataLoss,
    kUnauthenticated,
  };
  }  // namespace absl

  namespace absl {
  enum class StatusToStringMode : int {
    kWithNoExtraData = 0,
    kWithPayload = 1 << 0,
    kWithSourceLocation = 1 << 1,
    kWithEverything = ~kWithNoExtraData,
    kDefault = kWithPayload,
  };
  class Status {
   public:
    Status();
    template <typename Enum>
    Status(Enum code, std::string_view msg);
    Status(absl::StatusCode code, std::string_view msg,
           absl::SourceLocation loc = SourceLocation::current());
    Status(const Status& base_status, absl::SourceLocation loc);
    Status(Status&& base_status, absl::SourceLocation loc);
    ~Status() {}

    Status(const Status&);
    Status& operator=(const Status& x);

    Status(Status&&) noexcept;
    Status& operator=(Status&&);

    friend bool operator==(const Status&, const Status&);
    friend bool operator!=(const Status&, const Status&);

    bool ok() const { return true; }
    void CheckSuccess() const;
    void IgnoreError() const;
    int error_code() const;
    absl::Status ToCanonical() const;
    std::string ToString(
        StatusToStringMode m = StatusToStringMode::kDefault) const;
    void Update(const Status& new_status);
    void Update(Status&& new_status);

   private:
    void TestToStringSimplification() const {
      LOG(INFO) << this->ToString() << ToString();
    }
  };

  bool operator==(const Status& lhs, const Status& rhs);
  bool operator!=(const Status& lhs, const Status& rhs);

  Status OkStatus();
  Status InvalidArgumentError(char*);

  template <typename T>
  struct StatusOr;

  namespace internal_statusor {

  template <typename T, typename U, typename = void>
  struct HasConversionOperatorToStatusOr : std::false_type {};

  template <typename T, typename U>
  void test(char (*)[sizeof(std::declval<U>().operator absl::StatusOr<T>())]);

  template <typename T, typename U>
  struct HasConversionOperatorToStatusOr<T, U, decltype(test<T, U>(0))>
      : std::true_type {};

  template <typename T, typename U>
  using IsConstructibleOrConvertibleFromStatusOr =
      absl::disjunction<std::is_constructible<T, StatusOr<U>&>,
                        std::is_constructible<T, const StatusOr<U>&>,
                        std::is_constructible<T, StatusOr<U>&&>,
                        std::is_constructible<T, const StatusOr<U>&&>,
                        std::is_convertible<StatusOr<U>&, T>,
                        std::is_convertible<const StatusOr<U>&, T>,
                        std::is_convertible<StatusOr<U>&&, T>,
                        std::is_convertible<const StatusOr<U>&&, T>>;

  template <typename T, typename U>
  using IsConstructibleOrConvertibleOrAssignableFromStatusOr =
      absl::disjunction<IsConstructibleOrConvertibleFromStatusOr<T, U>,
                        std::is_assignable<T&, StatusOr<U>&>,
                        std::is_assignable<T&, const StatusOr<U>&>,
                        std::is_assignable<T&, StatusOr<U>&&>,
                        std::is_assignable<T&, const StatusOr<U>&&>>;

  template <typename T, typename U>
  struct IsDirectInitializationAmbiguous
      : public absl::conditional_t<
            std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                         U>::value,
            std::false_type,
            IsDirectInitializationAmbiguous<
                T, absl::remove_cv_t<absl::remove_reference_t<U>>>> {};

  template <typename T, typename V>
  struct IsDirectInitializationAmbiguous<T, absl::StatusOr<V>>
      : public IsConstructibleOrConvertibleFromStatusOr<T, V> {};

  template <typename T, typename U>
  using IsDirectInitializationValid = absl::disjunction<
      // Short circuits if T is basically U.
      std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
      absl::negation<absl::disjunction<
          std::is_same<absl::StatusOr<T>,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          std::is_same<absl::Status,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          std::is_same<absl::in_place_t,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          IsDirectInitializationAmbiguous<T, U>>>>;

  template <typename T, typename U>
  struct IsForwardingAssignmentAmbiguous
      : public absl::conditional_t<
            std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                         U>::value,
            std::false_type,
            IsForwardingAssignmentAmbiguous<
                T, absl::remove_cv_t<absl::remove_reference_t<U>>>> {};

  template <typename T, typename U>
  struct IsForwardingAssignmentAmbiguous<T, absl::StatusOr<U>>
      : public IsConstructibleOrConvertibleOrAssignableFromStatusOr<T, U> {};

  template <typename T, typename U>
  using IsForwardingAssignmentValid = absl::disjunction<
      // Short circuits if T is basically U.
      std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
      absl::negation<absl::disjunction<
          std::is_same<absl::StatusOr<T>,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          std::is_same<absl::Status,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          std::is_same<absl::in_place_t,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          IsForwardingAssignmentAmbiguous<T, U>>>>;

  template <typename T, typename U>
  using IsForwardingAssignmentValid = absl::disjunction<
      // Short circuits if T is basically U.
      std::is_same<T, absl::remove_cv_t<absl::remove_reference_t<U>>>,
      absl::negation<absl::disjunction<
          std::is_same<absl::StatusOr<T>,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          std::is_same<absl::Status,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          std::is_same<absl::in_place_t,
                       absl::remove_cv_t<absl::remove_reference_t<U>>>,
          IsForwardingAssignmentAmbiguous<T, U>>>>;

  template <typename T>
  struct OperatorBase {
    const T& value() const&;
    T& value() &;
    const T&& value() const&&;
    T&& value() &&;

    const T& operator*() const&;
    T& operator*() &;
    const T&& operator*() const&&;
    T&& operator*() &&;

    // To test that analyses are okay if there is a use of operator*
    // within this base class.
    const T* operator->() const { return __builtin_addressof(**this); }
    T* operator->() { return __builtin_addressof(**this); }
  };

  }  // namespace internal_statusor

  template <typename T>
  struct StatusOr : private internal_statusor::OperatorBase<T> {
    explicit StatusOr();

    StatusOr(const StatusOr&) = default;
    StatusOr& operator=(const StatusOr&) = default;

    StatusOr(StatusOr&&) = default;
    StatusOr& operator=(StatusOr&&) = default;

    template <
        typename U,
        absl::enable_if_t<
            absl::conjunction<
                absl::negation<std::is_same<T, U>>,
                std::is_constructible<T, const U&>,
                std::is_convertible<const U&, T>,
                absl::negation<
                    internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                        T, U>>>::value,
            int> = 0>
    StatusOr(const StatusOr<U>&);

    template <
        typename U,
        absl::enable_if_t<
            absl::conjunction<
                absl::negation<std::is_same<T, U>>,
                std::is_constructible<T, const U&>,
                absl::negation<std::is_convertible<const U&, T>>,
                absl::negation<
                    internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                        T, U>>>::value,
            int> = 0>
    explicit StatusOr(const StatusOr<U>&);

    template <
        typename U,
        absl::enable_if_t<
            absl::conjunction<
                absl::negation<std::is_same<T, U>>,
                std::is_constructible<T, U&&>, std::is_convertible<U&&, T>,
                absl::negation<
                    internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                        T, U>>>::value,
            int> = 0>
    StatusOr(StatusOr<U>&&);

    template <
        typename U,
        absl::enable_if_t<
            absl::conjunction<
                absl::negation<std::is_same<T, U>>,
                std::is_constructible<T, U&&>,
                absl::negation<std::is_convertible<U&&, T>>,
                absl::negation<
                    internal_statusor::IsConstructibleOrConvertibleFromStatusOr<
                        T, U>>>::value,
            int> = 0>
    explicit StatusOr(StatusOr<U>&&);

    template <
        typename U,
        absl::enable_if_t<
            absl::conjunction<
                absl::negation<std::is_same<T, U>>,
                std::is_constructible<T, const U&>,
                std::is_assignable<T, const U&>,
                absl::negation<
                    internal_statusor::
                        IsConstructibleOrConvertibleOrAssignableFromStatusOr<
                            T, U>>>::value,
            int> = 0>
    StatusOr& operator=(const StatusOr<U>&);

    template <
        typename U,
        absl::enable_if_t<
            absl::conjunction<
                absl::negation<std::is_same<T, U>>,
                std::is_constructible<T, U&&>, std::is_assignable<T, U&&>,
                absl::negation<
                    internal_statusor::
                        IsConstructibleOrConvertibleOrAssignableFromStatusOr<
                            T, U>>>::value,
            int> = 0>
    StatusOr& operator=(StatusOr<U>&&);

    template <typename U = absl::Status,
              absl::enable_if_t<
                  absl::conjunction<
                      std::is_convertible<U&&, absl::Status>,
                      std::is_constructible<absl::Status, U&&>,
                      absl::negation<
                          std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
                      absl::negation<std::is_same<absl::decay_t<U>, T>>,
                      absl::negation<
                          std::is_same<absl::decay_t<U>, absl::in_place_t>>,
                      absl::negation<
                          internal_statusor::HasConversionOperatorToStatusOr<
                              T, U&&>>>::value,
                  int> = 0>
    StatusOr(U&&);

    template <typename U = absl::Status,
              absl::enable_if_t<
                  absl::conjunction<
                      absl::negation<std::is_convertible<U&&, absl::Status>>,
                      std::is_constructible<absl::Status, U&&>,
                      absl::negation<
                          std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
                      absl::negation<std::is_same<absl::decay_t<U>, T>>,
                      absl::negation<
                          std::is_same<absl::decay_t<U>, absl::in_place_t>>,
                      absl::negation<
                          internal_statusor::HasConversionOperatorToStatusOr<
                              T, U&&>>>::value,
                  int> = 0>
    explicit StatusOr(U&&);

    template <typename U = absl::Status,
              absl::enable_if_t<
                  absl::conjunction<
                      std::is_convertible<U&&, absl::Status>,
                      std::is_constructible<absl::Status, U&&>,
                      absl::negation<
                          std::is_same<absl::decay_t<U>, absl::StatusOr<T>>>,
                      absl::negation<std::is_same<absl::decay_t<U>, T>>,
                      absl::negation<
                          std::is_same<absl::decay_t<U>, absl::in_place_t>>,
                      absl::negation<
                          internal_statusor::HasConversionOperatorToStatusOr<
                              T, U&&>>>::value,
                  int> = 0>
    StatusOr& operator=(U&&);

    template <
        typename U = T,
        typename = typename std::enable_if<absl::conjunction<
            std::is_constructible<T, U&&>, std::is_assignable<T&, U&&>,
            absl::disjunction<
                std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>, T>,
                absl::conjunction<
                    absl::negation<std::is_convertible<U&&, absl::Status>>,
                    absl::negation<
                        internal_statusor::HasConversionOperatorToStatusOr<
                            T, U&&>>>>,
            internal_statusor::IsForwardingAssignmentValid<T, U&&>>::value>::
            type>
    StatusOr& operator=(U&&);

    template <typename... Args>
    explicit StatusOr(absl::in_place_t, Args&&...);

    template <typename U, typename... Args>
    explicit StatusOr(absl::in_place_t, std::initializer_list<U>, Args&&...);

    template <
        typename U = T,
        absl::enable_if_t<
            absl::conjunction<
                internal_statusor::IsDirectInitializationValid<T, U&&>,
                std::is_constructible<T, U&&>, std::is_convertible<U&&, T>,
                absl::disjunction<
                    std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                                 T>,
                    absl::conjunction<
                        absl::negation<std::is_convertible<U&&, absl::Status>>,
                        absl::negation<
                            internal_statusor::HasConversionOperatorToStatusOr<
                                T, U&&>>>>>::value,
            int> = 0>
    StatusOr(U&&);

    template <
        typename U = T,
        absl::enable_if_t<
            absl::conjunction<
                internal_statusor::IsDirectInitializationValid<T, U&&>,
                absl::disjunction<
                    std::is_same<absl::remove_cv_t<absl::remove_reference_t<U>>,
                                 T>,
                    absl::conjunction<
                        absl::negation<
                            std::is_constructible<absl::Status, U&&>>,
                        absl::negation<
                            internal_statusor::HasConversionOperatorToStatusOr<
                                T, U&&>>>>,
                std::is_constructible<T, U&&>,
                absl::negation<std::is_convertible<U&&, T>>>::value,
            int> = 0>
    explicit StatusOr(U&&);

    bool ok() const;

    const Status& status() const& { return status_; }
    Status status() &&;

    using StatusOr::OperatorBase::value;

    const T& ValueOrDie() const&;
    T& ValueOrDie() &;
    const T&& ValueOrDie() const&&;
    T&& ValueOrDie() &&;

    using StatusOr::OperatorBase::operator*;

    using StatusOr::OperatorBase::operator->;

    template <typename U>
    T value_or(U&& default_value) const&;
    template <typename U>
    T value_or(U&& default_value) &&;

    template <typename... Args>
    T& emplace(Args&&... args);

    template <
        typename U, typename... Args,
        absl::enable_if_t<std::is_constructible<T, std::initializer_list<U>&,
                                                Args&&...>::value,
                          int> = 0>
    T& emplace(std::initializer_list<U> ilist, Args&&... args);

   private:
    absl::Status status_;
  };

  template <typename T>
  bool operator==(const StatusOr<T>& lhs, const StatusOr<T>& rhs);

  template <typename T>
  bool operator!=(const StatusOr<T>& lhs, const StatusOr<T>& rhs);

  }  // namespace absl

#endif  // STATUSOR_H_
    )cc";

TEST_P(UncheckedStatusOrAccessModelTest, NoStatusOrMention) {
  ExpectDiagnosticsFor(R"cc(
    void target() { "nop"; }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, NonExplicitInitialization) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"
    STATUSOR_INT target() {
      STATUSOR_INT x = Make<STATUSOR_INT>();
      return x.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToValue_NewLine) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      std::move(sor).value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToValueOrDie) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.ValueOrDie();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToValueOrDie) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      std::move(sor).ValueOrDie();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToOperatorStar) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      *sor;  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToOperatorStarSeparateLine) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      *  // [[unsafe]]
          sor;
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToOperatorStar) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      *std::move(sor);  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Lvalue_CallToOperatorArrow) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      void foo();
    };

    void target(absl::StatusOr<Foo> sor) {
      sor->foo();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest,
       UnwrapWithoutCheck_Rvalue_CallToOperatorArrow) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      void foo();
    };

    void target(absl::StatusOr<Foo> sor) {
      std::move(sor)->foo();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, UnwrapRvalueWithCheck) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (sor.ok()) std::move(sor).value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ParensInDeclInitExpr) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      auto sor = (Make<STATUSOR_INT>());
      if (sor.ok()) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ReferenceInDeclInitExpr) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      const STATUSOR_INT& GetStatusOrInt() const;
    };

    void target(Foo foo) {
      auto sor = foo.GetStatusOrInt();
      if (sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      STATUSOR_INT& GetStatusOrInt();
    };

    void target(Foo foo) {
      auto sor = foo.GetStatusOrInt();
      if (sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      STATUSOR_INT&& GetStatusOrInt() &&;
    };

    void target(Foo foo) {
      auto sor = std::move(foo).GetStatusOrInt();
      if (sor.ok()) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT sor) {
          if (sor.ok())
            sor.value();
          else
            sor.value();  // [[unsafe]]

          sor.value();  // [[unsafe]]
        }
      )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      if (auto sor = Make<STATUSOR_INT>(); sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, JoinSafeSafe) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT sor, bool b) {
          if (sor.ok()) {
            if (b)
              sor.value();
            else
              sor.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, JoinUnsafeUnsafe) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor, bool b) {
      if (b)
        sor.value();  // [[unsafe]]
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, InversedIfThenElse) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT sor) {
          if (!sor.ok())
            sor.value();  // [[unsafe]]
          else
            sor.value();

          sor.value();  // [[unsafe]]
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, DoubleInversedIfThenElse) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!!sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TripleInversedIfThenElse) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!!!sor.ok())
        sor.value();  // [[unsafe]]
      else
        sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (x.ok() && y.ok()) {
        x.value();

        y.value();
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() && y.ok()) {
            y.value();

            x.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsAndNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (x.ok() && !y.ok()) {
            x.value();

            y.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!x.ok() && !y.ok()) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(x.ok() && y.ok())) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(!x.ok() && y.ok())) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            y.value();

            x.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsAndNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(x.ok() && !y.ok())) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!(!x.ok() && !y.ok())) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (x.ok() || y.ok()) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsOrRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() || y.ok()) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_LhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (x.ok() || !y.ok()) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            y.value();

            x.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() || !y.ok()) {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          } else {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!(x.ok() || y.ok())) {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsOrRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(!x.ok() || y.ok())) {
            x.value();

            y.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_LhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!(x.ok() || !y.ok())) {
            y.value();

            x.value();  // [[unsafe]]
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, IfThenElse_Not_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      if (!(!x.ok() || !y.ok())) {
        x.value();

        y.value();
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TerminatingIfThenBranch) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) return;

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (sor.ok()) return;

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!x.ok() || !y.ok()) return;

          x.value();

          y.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TerminatingIfElseBranch) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (sor.ok()) {
      } else {
        return;
      }

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) {
      } else {
        return;
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TerminatingIfThenBranchInLoop) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (Make<bool>()) {
        if (!sor.ok()) continue;

        sor.value();
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (Make<bool>()) {
        if (!sor.ok()) break;

        sor.value();
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TernaryConditionalOperator) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.ok() ? sor.value() : 21;

      sor.ok() ? 21 : sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      !sor.ok() ? 21 : sor.value();

      !sor.ok() ? sor.value() : 21;  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor1, STATUSOR_INT sor2) {
      !((__builtin_expect(false || (!(sor1.ok() && sor2.ok())), false)))
          ? (void)0
          : (void)1;
      do {
        sor1.value();  // [[unsafe]]
        sor2.value();  // [[unsafe]]
      } while (true);
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (Make<bool>()) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!sor.ok()) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!!sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!!!sor.ok()) sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsAndRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (x.ok() && y.ok()) {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && y.ok()) y.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() && !y.ok()) x.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() && !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && !y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() && !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsAndRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && !y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() && !y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsAndNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && !y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() && !y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_LhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || !y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (x.ok() || !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || !y.ok()) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!x.ok() || !y.ok()) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsOrRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() || y.ok())) x.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(!x.ok() || y.ok())) y.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_LhsOrNotRhs) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || !y.ok())) x.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT x, STATUSOR_INT y) {
      while (!(x.ok() || !y.ok())) y.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_Not_NotLhsOrNotRhs) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (!(!x.ok() || !y.ok())) {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_AccessAfterStmt) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (sor.ok()) {
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!sor.ok()) {
      }

      sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_TerminatingBranch_Return) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (!sor.ok()) return;

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (sor.ok()) return;

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_TerminatingBranch_Continue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (Make<bool>()) {
        if (!sor.ok()) continue;

        sor.value();
      }

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, While_NestedIfWithBinaryCondition) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (Make<bool>()) {
            if (x.ok() && y.ok()) {
              x.value();

              y.value();
            }
          }
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          while (Make<bool>()) {
            if (!(!x.ok() || !y.ok())) {
              x.value();

              y.value();
            }
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, BuiltinExpect) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT x, STATUSOR_INT y) {
          if (!__builtin_expect(!x.ok() || __builtin_expect(!y.ok(), true), false)) {
            x.value();

            y.value();
          }
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CopyAssignment) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (sor.ok()) {
        sor = Make<STATUSOR_INT>();
        sor.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (!sor.ok()) return;

      sor = Make<STATUSOR_INT>();
      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      STATUSOR_INT x = Make<STATUSOR_INT>();
      if (x.ok()) {
        STATUSOR_INT y = x;
        x = Make<STATUSOR_INT>();

        y.value();

        x.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          STATUSOR_INT y = x;
          if (!y.ok()) return;

          x.value();

          y = Make<STATUSOR_INT>();
          x.value();
        }
      )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      STATUSOR_INT bar;
    };

    void target(Foo foo) {
      foo.bar = Make<STATUSOR_INT>();
      if (foo.bar.ok()) {
        foo.bar.value();

        foo.bar = Make<STATUSOR_INT>();
        foo.bar.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ShortCircuitingBinaryOperators) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = sor.ok() & sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = sor.ok() && sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = !sor.ok() && sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = sor.ok() || sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_BOOL sor) {
      bool b = !sor.ok() || sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b || sor.ok()) {
        do {
          sor.value();  // [[unsafe]]
        } while (true);
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (__builtin_expect(b || sor.ok(), false)) {
        do {
          sor.value();  // [[unsafe]]
        } while (false);
      }
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT sor1, STATUSOR_INT sor2) {
          while (sor1.ok() && sor2.ok()) sor1.value();
          while (sor1.ok() && sor2.ok()) sor2.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, References) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      STATUSOR_INT x = Make<STATUSOR_INT>();
      STATUSOR_INT& y = x;
      if (x.ok()) {
        x.value();

        y.value();
      } else {
        x.value();  // [[unsafe]]

        y.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          STATUSOR_INT& y = x;
          if (y.ok()) {
            x.value();

            y.value();
          } else {
            x.value();  // [[unsafe]]

            y.value();  // [[unsafe]]
          }
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          STATUSOR_INT& y = x;
          if (!y.ok()) return;

          x.value();

          y = Make<STATUSOR_INT>();
          x.value();  // [[unsafe]]
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target() {
          STATUSOR_INT x = Make<STATUSOR_INT>();
          const STATUSOR_INT& y = x;
          if (!y.ok()) return;

          y.value();

          x = Make<STATUSOR_INT>();
          y.value();  // [[unsafe]]
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, NoReturnAttribute) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    __attribute__((noreturn)) void f();

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) f();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void f();

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) f();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      __attribute__((noreturn)) ~Foo();
      void Bar();
    };

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) Foo().Bar();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      ~Foo();
      void Bar();
    };

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) Foo().Bar();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void f();
    __attribute__((noreturn)) void g();

    void target(STATUSOR_INT sor) {
      sor.ok() ? f() : g();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    __attribute__((noreturn)) void f();
    void g();

    void target(STATUSOR_INT sor) {
      !sor.ok() ? f() : g();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void f();
    void g();

    void target(STATUSOR_INT sor) {
      sor.ok() ? f() : g();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void terminate() __attribute__((noreturn));

    void target(STATUSOR_INT sor) {
      sor.value();  // [[unsafe]]
      terminate();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void terminate() __attribute__((noreturn));

    void target(STATUSOR_INT sor) {
      if (sor.ok()) sor.value();
      terminate();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void terminate() __attribute__((noreturn));

    struct Foo {
      ~Foo() __attribute__((noreturn));
    };

    void target() {
      auto sor = Make<absl::StatusOr<Foo>>();
      !(false || !(sor.ok())) ? (void)0 : terminate();
      sor.value();
      terminate();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, DeclInLoop) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      while (auto ok = sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    using BoolAlias = bool;

    void target(STATUSOR_INT sor) {
      while (BoolAlias ok = sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      while (Make<bool>()) {
        STATUSOR_INT sor = Make<STATUSOR_INT>();
        sor.value();  // [[unsafe]]
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    using StatusOrInt = STATUSOR_INT;

    void target() {
      while (Make<bool>()) {
        StatusOrInt sor = Make<STATUSOR_INT>();
        sor.value();  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, NonEvaluatedExprInCondition) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        bool unknown();

        void target(STATUSOR_INT sor) {
          if (unknown() && sor.ok()) sor.value();
          if (sor.ok() && unknown()) sor.value();
        }
      )cc");
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        bool unknown();

        void target(STATUSOR_INT sor) {
          if (!(!unknown() || !sor.ok())) sor.value();
          if (!(!sor.ok() || !unknown())) sor.value();
        }
      )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    bool unknown();

    void target(STATUSOR_INT sor) {
      if (unknown() || sor.ok()) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    bool unknown();

    void target(STATUSOR_INT sor) {
      if (sor.ok() || unknown()) sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CorrelatedBranches) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b || sor.ok()) {
        if (!b) sor.value();
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b && !sor.ok()) return;
      if (b) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (sor.ok()) b = true;
      if (b) sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      if (b) return;
      if (sor.ok()) b = true;
      if (b) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ConditionWithInitStmt) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      if (STATUSOR_INT sor = Make<STATUSOR_INT>(); sor.ok()) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      if (STATUSOR_INT sor = Make<STATUSOR_INT>(); !sor.ok())
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, DeadCode) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool b = false;
      if (b) sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool b;
      b = false;
      if (b) sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, TemporaryDestructors) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      sor.ok() ? sor.value() : Fatal().value();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      !sor.ok() ? Fatal().value() : sor.value();

      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      b ? 0 : sor.ok() ? sor.value() : Fatal().value();

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      for (int i = 0; i < 10; i++) {
        (b && sor.ok()) ? sor.value() : Fatal().value();

        if (b) sor.value();
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      for (int i = 0; i < 10; i++) {
        (b || !sor.ok()) ? Fatal().value() : 0;

        if (!b) sor.value();
      }
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b, STATUSOR_INT sor) {
      for (int i = 0; i < 10; i++) {
        (false || !(b && sor.ok())) ? Fatal().value() : 0;

        do {
          sor.value();
        } while (b);
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CheckMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      CHECK(sor.ok());
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      CHECK(!sor.ok());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, QcheckMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      QCHECK(sor.ok());
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      QCHECK(!sor.ok());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, CheckNeMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      CHECK_NE(sor.status(), absl::OkStatus());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, QcheckNeMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      QCHECK_NE(sor.status(), absl::OkStatus());
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, GlobalVars) {
  // The following examples are not sound as there could be opaque calls between
  // the ok() and the value() calls that change the StatusOr value.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    static STATUSOR_INT sor;

    void target() {
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      static STATUSOR_INT sor;
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;
    };

    void target(Foo foo) {
      if (foo.sor.ok())
        foo.sor.value();
      else
        foo.sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;
    };

    void target() {
      if (Foo::sor.ok())
        Foo::sor.value();
      else
        Foo::sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;

      static void target() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }
    };
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      static STATUSOR_INT sor;

      void target() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }
    };
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct S {
      static const int x = -1;
    };

    int target(S s) { return s.x; }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ReferenceReceivers) {
  // The following examples are not sound as there could be opaque calls between
  // the ok() and the value() calls that change the StatusOr value. However,
  // this is the behavior that users expect so it is here to stay.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT& sor) {
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      STATUSOR_INT& sor;
    };

    void target(Foo foo) {
      if (foo.sor.ok())
        foo.sor.value();
      else
        foo.sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Bar {
      STATUSOR_INT sor;
    };

    struct Foo {
      Bar& bar;
    };

    void target(Foo foo) {
      if (foo.bar.sor.ok())
        foo.bar.sor.value();
      else
        foo.bar.sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      STATUSOR_INT& sor;
    };

    void target(Foo& foo) {
      if (foo.sor.ok())
        foo.sor.value();
      else
        foo.sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Lambdas) {
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      [](STATUSOR_INT sor) {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }(Make<STATUSOR_INT>());
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      [sor]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      [&sor]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      [sor2 = sor]() {
        if (sor2.ok())
          sor2.value();
        else
          sor2.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      [&]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      [=]() {
        if (sor.ok())
          sor.value();
        else
          sor.value();  // [[unsafe]]
      }();
    }
  )cc");
  ExpectDiagnosticsForLambda(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct Foo {
      STATUSOR_INT sor;

      void target() {
        [this]() {
          if (sor.ok())
            sor.value();
          else
            sor.value();  // [[unsafe]]
        }();
      }
    };
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Status) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void foo();

    void target(STATUS s) {
      if (s.ok()) foo();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void foo();

    void target() {
      STATUS s = Make<STATUSOR_INT>().status();
      if (s.ok()) foo();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ExpectThatMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_THAT(sor, testing::status::IsOk());

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_THAT(sor.status(), testing::status::IsOk());

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      EXPECT_THAT(sor, testing::status::IsOk());

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ExpectOkMacro) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_OK(sor);

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      EXPECT_OK(sor.status());

      sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      EXPECT_OK(sor);

      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, BreadthFirstBlockTraversalLoop) {
  // Evaluating the CFG blocks of the code below in breadth-first order results
  // in an infinite loop. Each iteration of the while loop below results in a
  // new value being assigned to the storage location of sor1. However,
  // following a bread-first order of evaluation, downstream blocks will join
  // environments of different generations of predecessor blocks having distinct
  // values assigned to the sotrage location of sor1, resulting in not assigning
  // a value to the storage location of sor1 in successors. As iterations of the
  // analysis go, the state of the environment flips between having a value
  // assigned to the storage location of sor1 and not having a value assigned to
  // it. Since the evaluation of the copy constructor expression in bar(sor1)
  // depends on a value being assigned to sor1, the state of the environment
  // also flips between having a storage location assigned to the bar(sor1)
  // expression and not having a storage location assigned to it. This leads to
  // an infinite loop as the environment can't stabilize.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void foo(int, int);
    STATUSOR_INT bar(STATUSOR_INT);
    void baz(int);

    void target() {
      while (true) {
        STATUSOR_INT sor1 = Make<STATUSOR_INT>();
        if (sor1.ok()) {
          STATUSOR_INT sor2 = Make<STATUSOR_INT>();
          if (sor2.ok()) foo(sor1.value(), sor2.value());
        }

        STATUSOR_INT sor3 = bar(sor1);
        for (int i = 0; i < 5; i++) sor3 = bar(sor1);

        baz(sor3.value());  // [[unsafe]]
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, UsingDeclaration) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Goto) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
    label:
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
      goto label;
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
    label:
      if (!sor.ok()) goto label;
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      if (!sor.ok()) return;
      goto label;
    label:
      sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, JoinDistinctValues) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b) {
      STATUSOR_INT sor;
      if (b)
        sor = Make<STATUSOR_INT>();
      else
        sor = Make<STATUSOR_INT>();

      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b) {
      STATUSOR_INT sor;
      if (b) {
        sor = Make<STATUSOR_INT>();
        if (!sor.ok()) return;
      } else {
        sor = Make<STATUSOR_INT>();
        if (!sor.ok()) return;
      }
      sor.value();
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(bool b) {
      STATUSOR_INT sor;
      if (b) {
        sor = Make<STATUSOR_INT>();
        if (!sor.ok()) return;
      } else {
        sor = Make<STATUSOR_INT>();
      }
      sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, VarDeclInitExprWithoutValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      auto sor = Make<std::pair<int, STATUSOR_INT>>().second;
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      const auto& sor = Make<std::pair<int, STATUSOR_INT>>().second;
      if (sor.ok())
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, LValueToRValueCastOfChangingValue) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    bool foo();

    void target(bool b1) {
      STATUSOR_INT sor;
      if (b1)
        sor = Make<STATUSOR_INT>();
      else
        sor = Make<STATUSOR_INT>();

      do {
        const auto& b2 = foo();
        if (b2) break;

        sor.value();  // [[unsafe]]
      } while (true);
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, ConstructorInitializer) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    class target {
      target() : foo_(Make<STATUSOR_INT>().value()) {  // [[unsafe]]
      }
      int foo_;
    };
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, AssignStatusToBoolVar) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool ok = sor.ok();
      if (ok)
        sor.value();
      else
        sor.value();  // [[unsafe]]
    }
  )cc");
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target(STATUSOR_INT sor) {
      bool not_ok = !sor.ok();
      if (not_ok)
        sor.value();  // [[unsafe]]
      else
        sor.value();
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, StructuredBindings) {
  // Binding to a pair (which is actually a struct in the mock header).
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      const auto [sor, x] = Make<std::pair<STATUSOR_INT, int>>();
      if (sor.ok()) sor.value();
    }
  )cc");

  // Unsafe case.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      const auto [sor, x] = Make<std::pair<STATUSOR_INT, int>>();
      sor.value();  // [[unsafe]]
    }
  )cc");

  // As a reference.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      const auto& [sor, x] = Make<std::pair<STATUSOR_INT, int>>();
      if (sor.ok()) sor.value();
    }
  )cc");

  // Binding to a ref in a struct.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    struct S {
      STATUSOR_INT& sor;
      int i;
    };

    void target() {
      const auto& [sor, i] = Make<S>();
      if (sor.ok()) sor.value();
    }
  )cc");

  // In a loop.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      auto vals = Make<std::vector<std::pair<int, STATUSOR_INT>>>();
      for (const auto& [x, sor] : vals)
        if (sor.ok()) sor.value();
    }
  )cc");

  // Similar to the above, but InitExpr already has the storage initialized,
  // and bindings refer to them.
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    void target() {
      auto vals = Make<std::vector<std::pair<int, STATUSOR_INT>>>();
      for (const auto& p : vals) {
        const auto& [i, sor] = p;
        if (sor.ok()) sor.value();
      }
    }
  )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, AssignCompositeLogicExprToVar) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT sor, bool b) {
          bool c = sor.ok() && b;
          if (c) sor.value();
        }
      )cc");

  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        void target(STATUSOR_INT sor, bool b) {
          bool c = !(!sor.ok() || !b);
          if (c) sor.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, Subclass) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        class Bar;

        class Foo : public absl::StatusOr<Bar> {};

        void target(Foo opt) {
          opt.value();  // [[unsafe]]
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, SubclassStatus) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        class Foo : public STATUS {};

        void target(Foo opt) { opt.ok(); }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, SubclassOk) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        class Bar;

        class Foo : public absl::StatusOr<Bar> {};

        void target(Foo opt) {
          if (opt.ok()) opt.value();
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, SubclassOperator) {
  ExpectDiagnosticsFor(
      R"cc(
#include "unchecked_statusor_use_test_defs.h"

        class Bar;

        class Foo : public absl::StatusOr<Bar> {};

        void target(Foo opt) {
          *opt;  // [[unsafe]]
        }
      )cc");
}

TEST_P(UncheckedStatusOrAccessModelTest, GoodLambda) {
  ExpectDiagnosticsFor(R"cc(
#include "unchecked_statusor_use_test_defs.h"

    int target() {
      STATUSOR_INT sor = Make<STATUSOR_INT>();
      if (sor.ok()) return [&s = sor.value()] { return s; }();
      return 0;
    }
  )cc");
}

} // namespace

std::string
GetAliasMacros(UncheckedStatusOrAccessModelTestAliasKind AliasKind) {
  switch (AliasKind) {
  case UncheckedStatusOrAccessModelTestAliasKind::kUnaliased:
    return R"cc(
#define STATUSOR_INT ::absl::StatusOr<int>
#define STATUSOR_BOOL ::absl::StatusOr<bool>
#define STATUSOR_VOIDPTR ::absl::StatusOr<void*>
#define STATUS ::absl::Status
      )cc";
  case UncheckedStatusOrAccessModelTestAliasKind::kPartiallyAliased:
    return R"cc(
        template <typename T>
        using StatusOrAlias = ::absl::StatusOr<T>;
#define STATUSOR_INT StatusOrAlias<int>
#define STATUSOR_BOOL StatusOrAlias<bool>
#define STATUSOR_VOIDPTR StatusOrAlias<void*>
#define STATUS ::absl::Status
      )cc";
  case UncheckedStatusOrAccessModelTestAliasKind::kFullyAliased:
    return R"cc(
        using StatusOrIntAlias = ::absl::StatusOr<int>;
#define STATUSOR_INT StatusOrIntAlias
        using StatusOrBoolAlias = ::absl::StatusOr<bool>;
#define STATUSOR_BOOL StatusOrBoolAlias
        using StatusOrVoidPtrAlias = ::absl::StatusOr<void*>;
#define STATUSOR_VOIDPTR StatusOrVoidPtrAlias
        using StatusAlias = ::absl::Status;
#define STATUS StatusAlias
      )cc";
  }
  llvm_unreachable("Unknown alias kind.");
}

std::vector<std::pair<std::string, std::string>>
GetHeaders(UncheckedStatusOrAccessModelTestAliasKind AliasKind) {
  std::vector<std::pair<std::string, std::string>> Headers;
  Headers.emplace_back("absl_type_traits.h", kAbslTypeTraitsHeader);
  Headers.emplace_back("absl_defs.h", kAbslDefsHeader);
  Headers.emplace_back("std_type_traits.h", kStdTypeTraitsHeader);
  Headers.emplace_back("std_string_view.h", kStdStringViewHeader);
  Headers.emplace_back("std_char_traits.h", kStdCharTraitsHeader);
  Headers.emplace_back("string_defs.h", kStringDefsHeader);
  Headers.emplace_back("statusor_defs.h", kStatusOrDefsHeader);
  Headers.emplace_back("stdlib_defs.h", kStdLibDefsHeader);
  Headers.emplace_back("unchecked_statusor_use_test_defs.h",
                       R"cc(
#include "absl_defs.h"
#include "statusor_defs.h"

                             template <typename T>
                             T Make();

                             class Fatal {
                              public:
                               ~Fatal() __attribute__((noreturn));
                               int value();
                             };
                       )cc" +
                           GetAliasMacros(AliasKind));
  return Headers;
}
} // namespace clang::dataflow::statusor_model
