//===- UncheckedOptionalAccessModelTest.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// FIXME: Move this to clang/unittests/Analysis/FlowSensitive/Models.

#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "TestingSupport.h"
#include "clang/AST/ASTContext.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include <string>
#include <utility>
#include <vector>

using namespace clang;
using namespace dataflow;
using namespace test;

using ::testing::Pair;
using ::testing::UnorderedElementsAre;

// FIXME: Move header definitions in separate file(s).
static constexpr char StdTypeTraitsHeader[] = R"(
#ifndef TYPE_TRAITS_H
#define TYPE_TRAITS_H

namespace std {

typedef decltype(sizeof(char)) size_t;

template <typename T, T V>
struct integral_constant {
  static constexpr T value = V;
};

using true_type = integral_constant<bool, true>;
using false_type = integral_constant<bool, false>;

template< class T > struct remove_reference      {typedef T type;};
template< class T > struct remove_reference<T&>  {typedef T type;};
template< class T > struct remove_reference<T&&> {typedef T type;};

template <class T>
  using remove_reference_t = typename remove_reference<T>::type;

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

template <class T>
struct is_array : false_type {};

template <class T>
struct is_array<T[]> : true_type {};

template <class T, size_t N>
struct is_array<T[N]> : true_type {};

template <class>
struct is_function : false_type {};

template <class Ret, class... Args>
struct is_function<Ret(Args...)> : true_type {};

namespace detail {

template <class T>
struct type_identity {
  using type = T;
};  // or use type_identity (since C++20)

template <class T>
auto try_add_pointer(int) -> type_identity<typename remove_reference<T>::type*>;
template <class T>
auto try_add_pointer(...) -> type_identity<T>;

}  // namespace detail

template <class T>
struct add_pointer : decltype(detail::try_add_pointer<T>(0)) {};

template <bool B, class T, class F>
struct conditional {
  typedef T type;
};

template <class T, class F>
struct conditional<false, T, F> {
  typedef F type;
};

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
struct decay {
 private:
  typedef typename remove_reference<T>::type U;

 public:
  typedef typename conditional<
      is_array<U>::value, typename remove_extent<U>::type*,
      typename conditional<is_function<U>::value, typename add_pointer<U>::type,
                           typename remove_cv<U>::type>::type>::type type;
};

} // namespace std

#endif // TYPE_TRAITS_H
)";

static constexpr char StdUtilityHeader[] = R"(
#ifndef UTILITY_H
#define UTILITY_H

#include "std_type_traits.h"

namespace std {

template <typename T>
constexpr remove_reference_t<T>&& move(T&& x);

} // namespace std

#endif // UTILITY_H
)";

static constexpr char StdInitializerListHeader[] = R"(
#ifndef INITIALIZER_LIST_H
#define INITIALIZER_LIST_H

namespace std {

template <typename T>
class initializer_list {
 public:
  initializer_list() noexcept;
};

} // namespace std

#endif // INITIALIZER_LIST_H
)";

static constexpr char StdOptionalHeader[] = R"(
#include "std_initializer_list.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace std {

template <typename T>
class optional {
 public:
  constexpr optional() noexcept;

  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  const T* operator->() const;
  T* operator->();

  const T& value() const&;
  T& value() &;
  const T&& value() const&&;
  T&& value() &&;

  template <typename U>
  constexpr T value_or(U&& v) const&;
  template <typename U>
  T value_or(U&& v) &&;

  template <typename... Args>
  T& emplace(Args&&... args);

  template <typename U, typename... Args>
  T& emplace(std::initializer_list<U> ilist, Args&&... args);

  void reset() noexcept;

  constexpr explicit operator bool() const noexcept;
  constexpr bool has_value() const noexcept;
};

template <typename T>
constexpr optional<typename std::decay<T>::type> make_optional(T&& v);

template <typename T, typename... Args>
constexpr optional<T> make_optional(Args&&... args);

template <typename T, typename U, typename... Args>
constexpr optional<T> make_optional(std::initializer_list<U> il,
                                    Args&&... args);

} // namespace std
)";

static constexpr char AbslOptionalHeader[] = R"(
#include "std_initializer_list.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace absl {

template <typename T>
class optional {
 public:
  constexpr optional() noexcept;

  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  const T* operator->() const;
  T* operator->();

  const T& value() const&;
  T& value() &;
  const T&& value() const&&;
  T&& value() &&;

  template <typename U>
  constexpr T value_or(U&& v) const&;
  template <typename U>
  T value_or(U&& v) &&;

  template <typename... Args>
  T& emplace(Args&&... args);

  template <typename U, typename... Args>
  T& emplace(std::initializer_list<U> ilist, Args&&... args);

  void reset() noexcept;

  constexpr explicit operator bool() const noexcept;
  constexpr bool has_value() const noexcept;
};

template <typename T>
constexpr optional<typename std::decay<T>::type> make_optional(T&& v);

template <typename T, typename... Args>
constexpr optional<T> make_optional(Args&&... args);

template <typename T, typename U, typename... Args>
constexpr optional<T> make_optional(std::initializer_list<U> il,
                                    Args&&... args);

} // namespace absl
)";

static constexpr char BaseOptionalHeader[] = R"(
#include "std_initializer_list.h"
#include "std_type_traits.h"
#include "std_utility.h"

namespace base {

template <typename T>
class Optional {
 public:
  constexpr Optional() noexcept;

  const T& operator*() const&;
  T& operator*() &;
  const T&& operator*() const&&;
  T&& operator*() &&;

  const T* operator->() const;
  T* operator->();

  const T& value() const&;
  T& value() &;
  const T&& value() const&&;
  T&& value() &&;

  template <typename U>
  constexpr T value_or(U&& v) const&;
  template <typename U>
  T value_or(U&& v) &&;

  template <typename... Args>
  T& emplace(Args&&... args);

  template <typename U, typename... Args>
  T& emplace(std::initializer_list<U> ilist, Args&&... args);

  void reset() noexcept;

  constexpr explicit operator bool() const noexcept;
  constexpr bool has_value() const noexcept;
};

template <typename T>
constexpr Optional<typename std::decay<T>::type> make_optional(T&& v);

template <typename T, typename... Args>
constexpr Optional<T> make_optional(Args&&... args);

template <typename T, typename U, typename... Args>
constexpr Optional<T> make_optional(std::initializer_list<U> il,
                                    Args&&... args);

} // namespace base
)";

/// Converts `L` to string.
static std::string ConvertToString(const SourceLocationsLattice &L,
                                   const ASTContext &Ctx) {
  return L.getSourceLocations().empty() ? "safe"
                                        : "unsafe: " + DebugString(L, Ctx);
}

/// Replaces all occurrences of `Pattern` in `S` with `Replacement`.
static void ReplaceAllOccurrences(std::string &S, const std::string &Pattern,
                                  const std::string &Replacement) {
  size_t Pos = 0;
  while (true) {
    Pos = S.find(Pattern, Pos);
    if (Pos == std::string::npos)
      break;
    S.replace(Pos, Pattern.size(), Replacement);
  }
}

struct OptionalTypeIdentifier {
  std::string NamespaceName;
  std::string TypeName;
};

class UncheckedOptionalAccessTest
    : public ::testing::TestWithParam<OptionalTypeIdentifier> {
protected:
  template <typename LatticeChecksMatcher>
  void ExpectLatticeChecksFor(std::string SourceCode,
                              LatticeChecksMatcher MatchesLatticeChecks) {
    ExpectLatticeChecksFor(SourceCode, ast_matchers::hasName("target"),
                           MatchesLatticeChecks);
  }

private:
  template <typename FuncDeclMatcher, typename LatticeChecksMatcher>
  void ExpectLatticeChecksFor(std::string SourceCode,
                              FuncDeclMatcher FuncMatcher,
                              LatticeChecksMatcher MatchesLatticeChecks) {
    ReplaceAllOccurrences(SourceCode, "$ns", GetParam().NamespaceName);
    ReplaceAllOccurrences(SourceCode, "$optional", GetParam().TypeName);

    std::vector<std::pair<std::string, std::string>> Headers;
    Headers.emplace_back("std_initializer_list.h", StdInitializerListHeader);
    Headers.emplace_back("std_type_traits.h", StdTypeTraitsHeader);
    Headers.emplace_back("std_utility.h", StdUtilityHeader);
    Headers.emplace_back("std_optional.h", StdOptionalHeader);
    Headers.emplace_back("absl_optional.h", AbslOptionalHeader);
    Headers.emplace_back("base_optional.h", BaseOptionalHeader);
    Headers.emplace_back("unchecked_optional_access_test.h", R"(
      #include "absl_optional.h"
      #include "base_optional.h"
      #include "std_initializer_list.h"
      #include "std_optional.h"
      #include "std_utility.h"

      template <typename T>
      T Make();
    )");
    const tooling::FileContentMappings FileContents(Headers.begin(),
                                                    Headers.end());
    llvm::Error Error = checkDataflow<UncheckedOptionalAccessModel>(
        SourceCode, FuncMatcher,
        [](ASTContext &Ctx, Environment &) {
          return UncheckedOptionalAccessModel(Ctx);
        },
        [&MatchesLatticeChecks](
            llvm::ArrayRef<std::pair<
                std::string, DataflowAnalysisState<SourceLocationsLattice>>>
                CheckToLatticeMap,
            ASTContext &Ctx) {
          // FIXME: Consider using a matcher instead of translating
          // `CheckToLatticeMap` to `CheckToStringifiedLatticeMap`.
          std::vector<std::pair<std::string, std::string>>
              CheckToStringifiedLatticeMap;
          for (const auto &E : CheckToLatticeMap) {
            CheckToStringifiedLatticeMap.emplace_back(
                E.first, ConvertToString(E.second.Lattice, Ctx));
          }
          EXPECT_THAT(CheckToStringifiedLatticeMap, MatchesLatticeChecks);
        },
        {"-fsyntax-only", "-std=c++17", "-Wno-undefined-inline"}, FileContents);
    if (Error)
      FAIL() << llvm::toString(std::move(Error));
  }
};

INSTANTIATE_TEST_SUITE_P(
    UncheckedOptionalUseTestInst, UncheckedOptionalAccessTest,
    ::testing::Values(OptionalTypeIdentifier{"std", "optional"},
                      OptionalTypeIdentifier{"absl", "optional"},
                      OptionalTypeIdentifier{"base", "Optional"}),
    [](const ::testing::TestParamInfo<OptionalTypeIdentifier> &Info) {
      return Info.param.NamespaceName;
    });

TEST_P(UncheckedOptionalAccessTest, EmptyFunctionBody) {
  ExpectLatticeChecksFor(R"(
    void target() {
      (void)0;
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingValueNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      std::move(opt).value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingOperatorStarNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      *opt;
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:8")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      *std::move(opt);
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:8")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapUsingOperatorArrowNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      void foo();
    };

    void target($ns::$optional<Foo> opt) {
      opt->foo();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:9:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      void foo();
    };

    void target($ns::$optional<Foo> opt) {
      std::move(opt)->foo();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:9:7")));
}

TEST_P(UncheckedOptionalAccessTest, HasValueCheck) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt.has_value()) {
        opt.value();
        /*[[check]]*/
      }
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, OperatorBoolCheck) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      if (opt) {
        opt.value();
        /*[[check]]*/
      }
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, UnwrapFunctionCallResultNoCheck) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      Make<$ns::$optional<int>>().value();
      (void)0;
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> opt) {
      std::move(opt).value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:5:7")));
}

TEST_P(UncheckedOptionalAccessTest, DefaultConstructor) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:6:7")));
}

TEST_P(UncheckedOptionalAccessTest, MakeOptional) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::make_optional(0);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      Foo(int, int);
    };

    void target() {
      $ns::$optional<Foo> opt = $ns::make_optional<Foo>(21, 22);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    struct Foo {
      constexpr Foo(std::initializer_list<char>);
    };

    void target() {
      char a = 'a';
      $ns::$optional<Foo> opt = $ns::make_optional<Foo>({a});
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, ValueOr) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.value_or(0);
      (void)0;
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));
}

TEST_P(UncheckedOptionalAccessTest, Emplace) {
  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt;
      opt.emplace(0);
      opt.value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  ExpectLatticeChecksFor(R"(
    #include "unchecked_optional_access_test.h"

    void target($ns::$optional<int> *opt) {
      opt->emplace(0);
      opt->value();
      /*[[check]]*/
    }
  )",
                         UnorderedElementsAre(Pair("check", "safe")));

  // FIXME: Add tests that call `emplace` in conditional branches.
}

TEST_P(UncheckedOptionalAccessTest, Reset) {
  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> opt = $ns::make_optional(0);
      opt.reset();
      opt.value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:7:7")));

  ExpectLatticeChecksFor(
      R"(
    #include "unchecked_optional_access_test.h"

    void target() {
      $ns::$optional<int> *opt;
      *opt = $ns::make_optional(0);
      opt->reset();
      opt->value();
      /*[[check]]*/
    }
  )",
      UnorderedElementsAre(Pair("check", "unsafe: input.cc:8:7")));

  // FIXME: Add tests that call `reset` in conditional branches.
}

// FIXME: Add support for:
// - constructors (copy, move, non-standard)
// - assignment operators (default, copy, move, non-standard)
// - swap
// - invalidation (passing optional by non-const reference/pointer)
// - `value_or(nullptr) != nullptr`, `value_or(0) != 0`, `value_or("").empty()`
// - nested `optional` values
