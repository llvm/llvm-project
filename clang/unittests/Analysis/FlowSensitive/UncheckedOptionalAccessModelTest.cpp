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

static constexpr char StdTypeTraitsHeader[] = R"(
namespace std {

template< class T > struct remove_reference      {typedef T type;};
template< class T > struct remove_reference<T&>  {typedef T type;};
template< class T > struct remove_reference<T&&> {typedef T type;};

template <class T>
  using remove_reference_t = typename remove_reference<T>::type;

} // namespace std
)";

static constexpr char StdUtilityHeader[] = R"(
#include "std_type_traits.h"

namespace std {

template <typename T>
constexpr std::remove_reference_t<T>&& move(T&& x);

} // namespace std
)";

static constexpr char StdOptionalHeader[] = R"(
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

  constexpr bool has_value() const noexcept;
};

} // namespace std
)";

static constexpr char AbslOptionalHeader[] = R"(
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

  constexpr bool has_value() const noexcept;
};

} // namespace absl
)";

static constexpr char BaseOptionalHeader[] = R"(
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

  constexpr bool has_value() const noexcept;
};

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
    Headers.emplace_back("std_type_traits.h", StdTypeTraitsHeader);
    Headers.emplace_back("std_utility.h", StdUtilityHeader);
    Headers.emplace_back("std_optional.h", StdOptionalHeader);
    Headers.emplace_back("absl_optional.h", AbslOptionalHeader);
    Headers.emplace_back("base_optional.h", BaseOptionalHeader);
    Headers.emplace_back("unchecked_optional_access_test.h", R"(
      #include "absl_optional.h"
      #include "base_optional.h"
      #include "std_optional.h"
      #include "std_utility.h"
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

TEST_P(UncheckedOptionalAccessTest, UnwrapWithCheck) {
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

// FIXME: Add support for:
// - constructors (default, copy, move, non-standard)
// - assignment operators (default, copy, move, non-standard)
// - operator bool
// - emplace
// - reset
// - value_or
// - swap
// - make_optional
// - invalidation (passing optional by non-const reference/pointer)
